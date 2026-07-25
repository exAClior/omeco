//! Green-aware second-stage simulated annealing for contraction trees.
//!
//! Port of the RealifyTN paper's second-stage annealer. The Julia source
//! (`ComplexTN.jl/benchmarks/paper/greensa.jl`) is the algorithm
//! specification; it is never executed by the Rust campaign.
//!
//! # Objective
//!
//! A *realified* contraction executes complex tensors as paired real tensors.
//! Each internal tree node pays `2^(open indices) x factor`, where the factor
//! gauges the Re/Im ("green") structure of its two children:
//!
//! | left child | right child | factor | name  |
//! |------------|-------------|--------|-------|
//! | real       | real        | 1      | pass  |
//! | one green  | one green   | 2      | ride  |
//! | green      | green       | 3      | merge |
//!
//! A leaf is *green* when its tensor is structurally complex (nonzero
//! imaginary content); an internal node is green when either child is.
//! Green-blind search sets both factors to 1 (pure skeleton volume);
//! green-aware uses merge=3, ride=2 and may trade volume to move 3x merges
//! onto cheaper steps.
//!
//! # Algorithm (verbatim port)
//!
//! - Flat arrays (`left/right/parent/tensoridx`); per-node label-occurrence
//!   counts; precomputed label degrees `total`; a step's open-index count =
//!   labels with `0 < count < total` in either child. `refresh` recomputes a
//!   node from its two children in O(#labels).
//! - Move: associativity rotation at a random internal node, recomputing
//!   only the two touched nodes (O(1) incremental update, exact undo).
//! - Acceptance: `delta <= 0`, else with probability
//!   `exp(-(log2(cur+delta) - log2(cur)) / temp)` — the log2-ratio rule.
//! - Schedule: `temp = t0 * (t1/t0)^(k/nsteps)`, defaults `t0 = 1.0`,
//!   `t1 = 0.005`, `nsteps = 600_000`; multi-start over seeds `(42, 7, 2026)`
//!   from the same TreeSA start, best snapshot kept (`multi_anneal`).
//! - Polish (low-temperature) mode: `nsteps/2`, `t0 = 0.03`, `t1 = 0.002`,
//!   initialized from the best green-blind tree.
//! - Bookkeeping assert: after restoring the best snapshot, the reconstructed
//!   total cost matches the incremental running cost (`|dlog2| < 1e-6`).
//!
//! # Intentional deviations from the Julia source
//!
//! - Index base (0 vs 1) and RNG (`SmallRng` vs `MersenneTwister`): anneal
//!   trajectories differ, the algorithm and parameters do not.
//! - Open cost generalizes `2^nopen` to `2^(sum of log2 sizes of open
//!   labels)` so non-uniform bond dimensions work. Identical when every
//!   label has size 2 — which is the case for all of the campaign's
//!   quantum-circuit networks.
//! - Output labels (`iy`) count as open even when fully contained in one
//!   child subtree. The Julia condition misses those; the case is
//!   unreachable in the paper's scalar-output (`overlap`) networks, so
//!   costs there are identical either way.

use crate::eincode::{EinCode, NestedEinsum};
use crate::treesa::{build_label_map, convert_to_int_indices, optimize_treesa, TreeSA};
use crate::{CodeOptimizer, Label};
use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Sentinel for "no node" in the flat arrays (Julia uses 0; we are 0-indexed).
const NONE: usize = usize::MAX;

/// Configuration for the green-aware annealer.
///
/// Defaults are the paper's verbatim parameters: factors merge=3, ride=2;
/// `nsteps = 600_000`; schedule `t0 = 1.0 -> t1 = 0.005`; seeds
/// `(42, 7, 2026)`; polish schedule `t0 = 0.03 -> t1 = 0.002` at `nsteps/2`.
#[derive(Debug, Clone)]
pub struct GreenAnnealer {
    /// Cost factor when both children carry a green (Re/Im) leg. Paper: 3.0.
    pub merge_factor: f64,
    /// Cost factor when exactly one child carries a green leg. Paper: 2.0.
    pub ride_factor: f64,
    /// Annealing steps per seed.
    pub nsteps: usize,
    /// Initial temperature.
    pub t0: f64,
    /// Final temperature.
    pub t1: f64,
    /// Seeds for the multi-start anneal, all from the same starting tree.
    pub seeds: Vec<u64>,
    /// Polish-mode initial temperature (paper: 0.03).
    pub polish_t0: f64,
    /// Polish-mode final temperature (paper: 0.002).
    pub polish_t1: f64,
    /// TreeSA configuration producing the starting tree. The Julia pipeline
    /// initialized with `TreeSA(ntrials=1, niters=30)`; that is the default
    /// here. Campaign wiring overrides it with the campaign's TreeSA policy.
    pub initializer: TreeSA,
}

impl Default for GreenAnnealer {
    fn default() -> Self {
        Self {
            merge_factor: 3.0,
            ride_factor: 2.0,
            nsteps: 600_000,
            t0: 1.0,
            t1: 0.005,
            seeds: vec![42, 7, 2026],
            polish_t0: 0.03,
            polish_t1: 0.002,
            // Julia init: TreeSA(ntrials=1, niters=30) with default betas.
            initializer: TreeSA::default().with_ntrials(1).with_niters(30),
        }
    }
}

impl GreenAnnealer {
    /// Green-aware annealer (paper factors merge=3, ride=2).
    pub fn green_aware() -> Self {
        Self::default()
    }

    /// Green-blind annealer: factors (1, 1), i.e. pure skeleton-volume
    /// descent with the same engine and schedule. Produces the paper's
    /// "best real skeleton" baseline trees.
    pub fn green_blind() -> Self {
        Self {
            merge_factor: 1.0,
            ride_factor: 1.0,
            ..Self::default()
        }
    }

    /// Set the cost factors.
    pub fn with_factors(mut self, merge_factor: f64, ride_factor: f64) -> Self {
        self.merge_factor = merge_factor;
        self.ride_factor = ride_factor;
        self
    }

    /// Set the number of annealing steps per seed.
    pub fn with_nsteps(mut self, nsteps: usize) -> Self {
        self.nsteps = nsteps;
        self
    }

    /// Set the temperature schedule endpoints.
    pub fn with_schedule(mut self, t0: f64, t1: f64) -> Self {
        self.t0 = t0;
        self.t1 = t1;
        self
    }

    /// Set the multi-start seed list.
    pub fn with_seeds(mut self, seeds: Vec<u64>) -> Self {
        self.seeds = seeds;
        self
    }

    /// Set the TreeSA initializer for the starting tree.
    pub fn with_initializer(mut self, initializer: TreeSA) -> Self {
        self.initializer = initializer;
        self
    }

    /// Run TreeSA init, then the green multi-start anneal, returning the
    /// annealed tree plus a cost report.
    ///
    /// `is_complex[t]` must say whether tensor `t` is structurally complex
    /// (carries a green leg).
    pub fn optimize_with_green<L: Label>(
        &self,
        code: &EinCode<L>,
        size_dict: &HashMap<L, usize>,
        is_complex: &[bool],
    ) -> Option<(NestedEinsum<L>, GreenReport)> {
        assert_eq!(
            is_complex.len(),
            code.num_tensors(),
            "is_complex length must match the number of tensors"
        );
        let start = optimize_treesa(code, size_dict, &self.initializer)?;
        if start.is_leaf() {
            return Some((
                start,
                GreenReport {
                    start_cost: 0.0,
                    final_cost: 0.0,
                    pass_volume: 0.0,
                    ride_volume: 0.0,
                    merge_volume: 0.0,
                },
            ));
        }
        let (label_map, labels) = build_label_map(code);
        let int_ixs = convert_to_int_indices(&code.ixs, &label_map);
        let int_iy: Vec<usize> = code.iy.iter().map(|l| label_map[l]).collect();
        let log2_sizes: Vec<f64> = labels
            .iter()
            .map(|l| (size_dict[l] as f64).log2())
            .collect();
        let mut tree = GreenTree::from_nested(
            &start,
            int_ixs,
            int_iy,
            log2_sizes,
            is_complex.to_vec(),
            self.merge_factor,
            self.ride_factor,
        );
        let start_cost = tree.total_cost();
        let start_snap = tree.snapshot();
        let final_cost = tree.multi_anneal(
            &start_snap,
            self.nsteps,
            self.t0,
            self.t1,
            &self.seeds,
        );
        let (pass_volume, ride_volume, merge_volume) = tree.breakdown();
        let nested = tree.to_nested(&code.ixs, &code.iy, &labels);
        Some((
            nested,
            GreenReport {
                start_cost,
                final_cost,
                pass_volume,
                ride_volume,
                merge_volume,
            },
        ))
    }
}

impl CodeOptimizer for GreenAnnealer {
    /// The `CodeOptimizer` trait carries no structural-complexity
    /// information, so every tensor is treated as complex. Every internal
    /// node then pays the (constant) merge factor and the anneal reduces to
    /// pure volume descent from the TreeSA start. Callers that know which
    /// tensors are complex should use [`GreenAnnealer::optimize_with_green`].
    fn optimize<L: Label>(
        &self,
        code: &EinCode<L>,
        size_dict: &HashMap<L, usize>,
    ) -> Option<NestedEinsum<L>> {
        let is_complex = vec![true; code.num_tensors()];
        self.optimize_with_green(code, size_dict, &is_complex)
            .map(|(nested, _)| nested)
    }
}

/// Cost report for one green anneal.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GreenReport {
    /// Objective value of the starting tree (with the annealer's factors).
    pub start_cost: f64,
    /// Objective value of the best tree found (with the annealer's factors).
    pub final_cost: f64,
    /// Total real volume of pass (real-real) steps in the final tree.
    pub pass_volume: f64,
    /// Total real volume of ride (one-green) steps in the final tree.
    pub ride_volume: f64,
    /// Total real volume of merge (green-green) steps in the final tree.
    pub merge_volume: f64,
}

/// Structure-only snapshot of a [`GreenTree`] (Julia's `snapshot`).
///
/// Snapshots are valid across `GreenTree` instances built from the *same*
/// starting `NestedEinsum` (flat node indices are a deterministic function
/// of that tree). This is how the polish pass inherits the best green-blind
/// skeleton.
#[derive(Debug, Clone)]
pub struct TreeSnapshot {
    left: Vec<usize>,
    right: Vec<usize>,
    parent: Vec<usize>,
    root: usize,
}

/// Saved state for exact undo of one rotation (Julia's `old` tuple).
/// Counts of node `c` live in [`RotationScratch`].
#[derive(Debug, Clone)]
pub struct Rotation {
    /// Change in total step cost (`dnew - dold` over the two touched nodes).
    pub delta: f64,
    n: usize,
    c: usize,
    x: usize,
    keep: usize,
    mov: usize,
    old_left_n: usize,
    old_right_n: usize,
    old_left_c: usize,
    old_right_c: usize,
    old_parent_x: usize,
    old_parent_keep: usize,
    old_parent_mov: usize,
    old_green_c: bool,
    old_step_c: f64,
    old_step_n: f64,
    old_real_c: f64,
    old_real_n: f64,
}

/// Reusable scratch space for rotations (avoids a per-step allocation).
///
/// Holds the saved count column of node `c`. An [`GreenTree::undo`] must be
/// applied before the next [`GreenTree::try_rotation`] overwrites it.
#[derive(Debug, Clone)]
pub struct RotationScratch {
    old_counts_c: Vec<i16>,
}

impl RotationScratch {
    /// Create scratch space for a tree with `nlabels` distinct labels.
    pub fn new(nlabels: usize) -> Self {
        Self {
            old_counts_c: Vec::with_capacity(nlabels),
        }
    }
}

/// A binary contraction tree as flat arrays, with incremental green-aware
/// cost bookkeeping. Direct port of the Julia `Tree` struct.
///
/// All label references are integer indices into `0..nlabels`.
#[derive(Clone)]
pub struct GreenTree {
    left: Vec<usize>,
    right: Vec<usize>,
    parent: Vec<usize>,
    tensoridx: Vec<usize>,
    /// Node-major count matrix: `counts[i * nlabels + l]` = occurrences of
    /// label `l` in the subtree rooted at node `i`.
    counts: Vec<i16>,
    green: Vec<bool>,
    stepcost: Vec<f64>,
    realcost: Vec<f64>,
    root: usize,
    /// Total occurrences of each label across all input tensors.
    total: Vec<i16>,
    nlabels: usize,
    ixs: Vec<Vec<usize>>,
    is_output: Vec<bool>,
    log2_sizes: Vec<f64>,
    is_complex: Vec<bool>,
    merge_factor: f64,
    ride_factor: f64,
}

impl GreenTree {
    /// Build a flat tree from a binary [`NestedEinsum`] (structure only; the
    /// stored einsum operations are ignored and recomputed on export).
    ///
    /// * `ixs` — input tensor labels (integer domain), one entry per tensor.
    /// * `iy` — output labels (integer domain).
    /// * `log2_sizes` — log2 of each label's dimension (all `1.0` for the
    ///   campaign's size-2 quantum indices).
    /// * `is_complex` — per-tensor structural complexity (green legs).
    /// * `merge_factor`, `ride_factor` — green cost factors.
    pub fn from_nested<L: Label>(
        nested: &NestedEinsum<L>,
        ixs: Vec<Vec<usize>>,
        iy: Vec<usize>,
        log2_sizes: Vec<f64>,
        is_complex: Vec<bool>,
        merge_factor: f64,
        ride_factor: f64,
    ) -> Self {
        let nlabels = log2_sizes.len();
        assert_eq!(ixs.len(), is_complex.len());
        let mut total = vec![0i16; nlabels];
        for ix in &ixs {
            for &l in ix {
                total[l] += 1;
            }
        }
        let mut is_output = vec![false; nlabels];
        for &l in &iy {
            is_output[l] = true;
        }
        let mut tree = Self {
            left: Vec::new(),
            right: Vec::new(),
            parent: Vec::new(),
            tensoridx: Vec::new(),
            counts: Vec::new(),
            green: Vec::new(),
            stepcost: Vec::new(),
            realcost: Vec::new(),
            root: NONE,
            total,
            nlabels,
            ixs,
            is_output,
            log2_sizes,
            is_complex,
            merge_factor,
            ride_factor,
        };
        let root = tree.push_subtree(nested);
        tree.root = root;
        let nnodes = tree.left.len();
        tree.counts = vec![0i16; nlabels * nnodes];
        tree.green = vec![false; nnodes];
        tree.stepcost = vec![0.0; nnodes];
        tree.realcost = vec![0.0; nnodes];
        tree.recompute_from_structure();
        tree
    }

    /// Recursive pre-order push (matches Julia's `build`): parents always
    /// receive smaller indices than the children pushed afterwards.
    fn push_subtree<L: Label>(&mut self, x: &NestedEinsum<L>) -> usize {
        let me = self.left.len();
        self.left.push(NONE);
        self.right.push(NONE);
        self.parent.push(NONE);
        self.tensoridx.push(NONE);
        match x {
            NestedEinsum::Leaf { tensor_index } => {
                self.tensoridx[me] = *tensor_index;
            }
            NestedEinsum::Node { args, .. } => {
                assert!(args.len() == 2, "green annealer requires a binary tree");
                let a = self.push_subtree(&args[0]);
                let b = self.push_subtree(&args[1]);
                self.left[me] = a;
                self.right[me] = b;
                self.parent[a] = me;
                self.parent[b] = me;
            }
        }
        me
    }

    /// Number of nodes (internal + leaves).
    pub fn nnodes(&self) -> usize {
        self.left.len()
    }

    /// Whether node `i` is a leaf.
    #[inline]
    pub fn is_leaf(&self, i: usize) -> bool {
        self.left[i] == NONE
    }

    /// Root node index.
    pub fn root(&self) -> usize {
        self.root
    }

    /// Total objective: sum of `stepcost` over internal nodes.
    pub fn total_cost(&self) -> f64 {
        self.stepcost.iter().sum()
    }

    /// Total green-free volume: sum of `realcost` over internal nodes.
    pub fn total_real(&self) -> f64 {
        self.realcost.iter().sum()
    }

    /// Real volume split by step kind: `(pass, ride, merge)`.
    pub fn breakdown(&self) -> (f64, f64, f64) {
        let (mut pass, mut ride, mut merge) = (0.0, 0.0, 0.0);
        for i in 0..self.nnodes() {
            if self.is_leaf(i) {
                continue;
            }
            let ga = self.green[self.left[i]];
            let gb = self.green[self.right[i]];
            if ga && gb {
                merge += self.realcost[i];
            } else if ga || gb {
                ride += self.realcost[i];
            } else {
                pass += self.realcost[i];
            }
        }
        (pass, ride, merge)
    }

    /// Recompute every node's counts/green/costs from the current structure
    /// (Julia's `initnode!` from the root). Uses an explicit post-order
    /// traversal: after rotations, node indices no longer respect build
    /// order, so index-order sweeps are not a valid post-order.
    pub fn recompute_from_structure(&mut self) {
        self.counts.fill(0);
        self.green.fill(false);
        self.stepcost.fill(0.0);
        self.realcost.fill(0.0);
        let mut stack = vec![(self.root, false)];
        while let Some((i, children_done)) = stack.pop() {
            if self.is_leaf(i) {
                let t = self.tensoridx[i];
                for &l in &self.ixs[t] {
                    self.counts[i * self.nlabels + l] += 1;
                }
                self.green[i] = self.is_complex[t];
            } else if children_done {
                self.refresh(i);
            } else {
                stack.push((i, true));
                stack.push((self.left[i], false));
                stack.push((self.right[i], false));
            }
        }
    }

    /// Recompute node `i` from its two children in O(#labels).
    #[inline]
    fn refresh(&mut self, i: usize) {
        let a = self.left[i];
        let b = self.right[i];
        let nl = self.nlabels;
        let (ba, bb, bi) = (a * nl, b * nl, i * nl);
        let mut open = 0.0f64;
        for l in 0..nl {
            let ca = self.counts[ba + l];
            let cb = self.counts[bb + l];
            self.counts[bi + l] = ca + cb;
            let tot = self.total[l];
            if (ca > 0 && ca < tot)
                || (cb > 0 && cb < tot)
                || (self.is_output[l] && (ca > 0 || cb > 0))
            {
                open += self.log2_sizes[l];
            }
        }
        let ga = self.green[a];
        let gb = self.green[b];
        self.green[i] = ga || gb;
        let rc = open.exp2();
        let factor = if ga && gb {
            self.merge_factor
        } else if ga || gb {
            self.ride_factor
        } else {
            1.0
        };
        self.realcost[i] = rc;
        self.stepcost[i] = rc * factor;
    }

    /// Attempt one associativity rotation at a random internal node.
    ///
    /// Returns `None` when the sampled node is a leaf or has no internal
    /// child (Julia: `return nothing`). Otherwise rewires the tree,
    /// recomputes the two touched nodes, and returns the cost delta plus
    /// the undo record. Apply [`Self::undo`] to revert.
    pub fn try_rotation<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        scratch: &mut RotationScratch,
    ) -> Option<Rotation> {
        let n = rng.random_range(0..self.nnodes());
        if self.is_leaf(n) {
            return None;
        }
        let a = self.left[n];
        let b = self.right[n];
        let mut cands = [NONE; 2];
        let mut ncand = 0;
        if !self.is_leaf(a) {
            cands[ncand] = a;
            ncand += 1;
        }
        if !self.is_leaf(b) {
            cands[ncand] = b;
            ncand += 1;
        }
        if ncand == 0 {
            return None;
        }
        let c = cands[rng.random_range(0..ncand)];
        let x = if c == a { b } else { a };
        let bb = self.left[c];
        let cc = self.right[c];
        let keep = if rng.random::<bool>() { bb } else { cc };
        let mov = if keep == bb { cc } else { bb };

        let rot = Rotation {
            delta: 0.0,
            n,
            c,
            x,
            keep,
            mov,
            old_left_n: self.left[n],
            old_right_n: self.right[n],
            old_left_c: self.left[c],
            old_right_c: self.right[c],
            old_parent_x: self.parent[x],
            old_parent_keep: self.parent[keep],
            old_parent_mov: self.parent[mov],
            old_green_c: self.green[c],
            old_step_c: self.stepcost[c],
            old_step_n: self.stepcost[n],
            old_real_c: self.realcost[c],
            old_real_n: self.realcost[n],
        };
        let nl = self.nlabels;
        scratch.old_counts_c.clear();
        scratch
            .old_counts_c
            .extend_from_slice(&self.counts[c * nl..(c + 1) * nl]);

        // Before: n = (c, x) with c = (keep, mov) [child order may vary].
        // After:  c = (x, keep); n = (c, mov).
        // Node n keeps its leaf set, so only c and n need recomputation.
        self.left[c] = x;
        self.right[c] = keep;
        self.parent[x] = c;
        self.parent[keep] = c;
        self.left[n] = c;
        self.right[n] = mov;
        self.parent[mov] = n;
        self.parent[c] = n;

        let dold = rot.old_step_c + rot.old_step_n;
        self.refresh(c);
        self.refresh(n);
        let dnew = self.stepcost[c] + self.stepcost[n];
        Some(Rotation {
            delta: dnew - dold,
            ..rot
        })
    }

    /// Revert a rotation exactly. Must be called before the next
    /// [`Self::try_rotation`] (the scratch buffer holds the saved counts).
    pub fn undo(&mut self, rot: &Rotation, scratch: &mut RotationScratch) {
        self.left[rot.n] = rot.old_left_n;
        self.right[rot.n] = rot.old_right_n;
        self.left[rot.c] = rot.old_left_c;
        self.right[rot.c] = rot.old_right_c;
        self.parent[rot.x] = rot.old_parent_x;
        self.parent[rot.keep] = rot.old_parent_keep;
        self.parent[rot.mov] = rot.old_parent_mov;
        let nl = self.nlabels;
        self.counts[rot.c * nl..(rot.c + 1) * nl].copy_from_slice(&scratch.old_counts_c);
        self.green[rot.c] = rot.old_green_c;
        self.stepcost[rot.c] = rot.old_step_c;
        self.stepcost[rot.n] = rot.old_step_n;
        self.realcost[rot.c] = rot.old_real_c;
        self.realcost[rot.n] = rot.old_real_n;
    }

    /// Structure-only snapshot (cheap: three index vectors + root).
    pub fn snapshot(&self) -> TreeSnapshot {
        TreeSnapshot {
            left: self.left.clone(),
            right: self.right.clone(),
            parent: self.parent.clone(),
            root: self.root,
        }
    }

    /// Restore a snapshot and rebuild all bookkeeping from the structure.
    pub fn restore(&mut self, snap: &TreeSnapshot) {
        self.left.clone_from(&snap.left);
        self.right.clone_from(&snap.right);
        self.parent.clone_from(&snap.parent);
        self.root = snap.root;
        self.recompute_from_structure();
    }

    /// One annealing run: geometric schedule `t0 -> t1` over `nsteps`,
    /// log2-ratio Metropolis acceptance, best snapshot restored at the end.
    /// Returns the best total cost seen.
    pub fn anneal(&mut self, nsteps: usize, t0: f64, t1: f64, seed: u64) -> f64 {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let mut scratch = RotationScratch::new(self.nlabels);
        let mut cur = self.total_cost();
        let mut best = cur;
        let mut best_snap = self.snapshot();
        for k in 1..=nsteps {
            let temp = t0 * (t1 / t0).powf(k as f64 / nsteps as f64);
            let Some(rot) = self.try_rotation(&mut rng, &mut scratch) else {
                continue;
            };
            if rot.delta <= 0.0
                || rng.random::<f64>()
                    < (-((cur + rot.delta).log2() - cur.log2()) / temp).exp()
            {
                cur += rot.delta;
                if cur < best {
                    best = cur;
                    best_snap = self.snapshot();
                }
            } else {
                self.undo(&rot, &mut scratch);
            }
        }
        self.restore(&best_snap);
        let tc = self.total_cost();
        assert!(
            (tc.log2() - best.log2()).abs() < 1e-6,
            "bookkeeping drift: {tc} vs {best}"
        );
        tc
    }

    /// Multi-start anneal (Julia's `multi_anneal!`): every seed restarts
    /// from `start`; the best tree across seeds is kept.
    pub fn multi_anneal(
        &mut self,
        start: &TreeSnapshot,
        nsteps: usize,
        t0: f64,
        t1: f64,
        seeds: &[u64],
    ) -> f64 {
        let mut bestv = f64::INFINITY;
        let mut best_snap = start.clone();
        for &seed in seeds {
            self.restore(start);
            let v = self.anneal(nsteps, t0, t1, seed);
            if v < bestv {
                bestv = v;
                best_snap = self.snapshot();
            }
        }
        self.restore(&best_snap);
        bestv
    }

    /// Export the current tree as a [`NestedEinsum`] with recomputed einsum
    /// operations. Child outputs feed parent inputs verbatim, so the result
    /// passes exact label-order validation; the root keeps `iy` exactly.
    ///
    /// * `original_ixs` — original input labels per tensor (label domain).
    /// * `iy` — original output labels (label domain); used verbatim at root.
    /// * `inverse_map` — integer label index -> original label.
    pub fn to_nested<L: Label>(
        &self,
        original_ixs: &[Vec<L>],
        iy: &[L],
        inverse_map: &[L],
    ) -> NestedEinsum<L> {
        let label_map: HashMap<L, usize> = inverse_map
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, l)| (l, i))
            .collect();
        self.build_nested(self.root, original_ixs, iy, &label_map)
            .0
    }

    fn build_nested<L: Label>(
        &self,
        i: usize,
        original_ixs: &[Vec<L>],
        iy: &[L],
        label_map: &HashMap<L, usize>,
    ) -> (NestedEinsum<L>, Vec<L>) {
        if self.is_leaf(i) {
            let t = self.tensoridx[i];
            return (NestedEinsum::leaf(t), original_ixs[t].clone());
        }
        let (left_tree, left_out) =
            self.build_nested(self.left[i], original_ixs, iy, label_map);
        let (right_tree, right_out) =
            self.build_nested(self.right[i], original_ixs, iy, label_map);
        let node_iy: Vec<L> = if i == self.root {
            iy.to_vec()
        } else {
            // Open labels of subtree i, in left-then-right child order.
            let mut seen = vec![false; self.nlabels];
            let mut out = Vec::new();
            for l in left_out.iter().chain(right_out.iter()) {
                let k = label_map[l];
                if !seen[k] && self.is_open_at(i, k) {
                    seen[k] = true;
                    out.push(l.clone());
                }
            }
            out
        };
        let eins = EinCode::new(vec![left_out, right_out], node_iy.clone());
        (NestedEinsum::node(vec![left_tree, right_tree], eins), node_iy)
    }

    /// Whether label `l` is an output of subtree `i`: present in the subtree
    /// and either present elsewhere in the network or part of the final
    /// output.
    #[inline]
    fn is_open_at(&self, i: usize, l: usize) -> bool {
        let c = self.counts[i * self.nlabels + l];
        c > 0 && (c < self.total[l] || self.is_output[l])
    }
}

/// Outcome of the paper's three-pass optimization pipeline (`fig-pipe`).
#[derive(Debug, Clone)]
pub struct GreenPipeOutcome<L: Label> {
    /// The shared TreeSA starting tree.
    pub start: NestedEinsum<L>,
    /// Pass 1 result: green-blind anneal (factors 1, 1) — the best real
    /// skeleton found by any pass.
    pub best_blind: NestedEinsum<L>,
    /// Pass 3 result: low-temperature green-aware polish of `best_blind`.
    pub polished: NestedEinsum<L>,
    /// Pass 2 result: full green-aware anneal from the TreeSA start.
    pub full_anneal: NestedEinsum<L>,
    /// Aware-factor cost of the TreeSA start (Julia's `blind_green`): what a
    /// green-unaware tree costs once realified.
    pub green_blind_cost: f64,
    /// Pass 1 objective value (factors 1, 1): best real skeleton volume.
    pub best_real_cost: f64,
    /// Aware-factor cost of the pass-1 tree with no reoptimization
    /// (Julia's `converted`).
    pub convert_only_cost: f64,
    /// Pass 3 objective value (Julia's `quench_green`).
    pub polished_cost: f64,
    /// Pass 2 objective value (Julia's `best_green`).
    pub full_anneal_cost: f64,
    /// Green-free volume of the TreeSA start (`2^tc` of the starting tree).
    pub start_volume: f64,
    /// Pass-step real volume of the full-anneal tree.
    pub full_anneal_pass_volume: f64,
    /// Ride-step real volume of the full-anneal tree.
    pub full_anneal_ride_volume: f64,
    /// Merge-step real volume of the full-anneal tree.
    pub full_anneal_merge_volume: f64,
    /// Wall seconds per stage: `(treesa_init, blind_anneal, aware_anneal, polish)`.
    pub seconds: (f64, f64, f64, f64),
}

/// The paper's three-pass pipeline from one TreeSA start:
///
/// 1. Green-blind anneal (factors 1, 1) — best real skeleton baseline.
/// 2. Full green-aware anneal from the same start.
/// 3. Convert the pass-1 tree (no reoptimization) and apply a
///    low-temperature green-aware polish (`nsteps/2`, `polish_t0`,
///    `polish_t1`).
///
/// All passes share one starting tree, so the four costs are directly
/// comparable (the `fig-pipe` flat-landscape dataset).
pub fn green_pipeline<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    is_complex: &[bool],
    config: &GreenAnnealer,
) -> Option<GreenPipeOutcome<L>> {
    assert_eq!(
        is_complex.len(),
        code.num_tensors(),
        "is_complex length must match the number of tensors"
    );
    let t_start = Instant::now();
    let start = optimize_treesa(code, size_dict, &config.initializer)?;
    let init_seconds = t_start.elapsed().as_secs_f64();
    if start.is_leaf() {
        return Some(GreenPipeOutcome {
            start: start.clone(),
            best_blind: start.clone(),
            polished: start.clone(),
            full_anneal: start,
            green_blind_cost: 0.0,
            best_real_cost: 0.0,
            convert_only_cost: 0.0,
            polished_cost: 0.0,
            full_anneal_cost: 0.0,
            start_volume: 0.0,
            full_anneal_pass_volume: 0.0,
            full_anneal_ride_volume: 0.0,
            full_anneal_merge_volume: 0.0,
            seconds: (init_seconds, 0.0, 0.0, 0.0),
        });
    }
    let (label_map, labels) = build_label_map(code);
    let int_ixs = convert_to_int_indices(&code.ixs, &label_map);
    let int_iy: Vec<usize> = code.iy.iter().map(|l| label_map[l]).collect();
    let log2_sizes: Vec<f64> = labels
        .iter()
        .map(|l| (size_dict[l] as f64).log2())
        .collect();

    // Pass 1: anneal the real skeleton (factors 1, 1) — fair baseline.
    let t_blind = Instant::now();
    let mut blind_tree = GreenTree::from_nested(
        &start,
        int_ixs.clone(),
        int_iy.clone(),
        log2_sizes.clone(),
        is_complex.to_vec(),
        1.0,
        1.0,
    );
    let start_volume = blind_tree.total_real();
    let start_snap = blind_tree.snapshot();
    let best_real_cost = blind_tree.multi_anneal(
        &start_snap,
        config.nsteps,
        config.t0,
        config.t1,
        &config.seeds,
    );
    let blind_snap = blind_tree.snapshot();
    let blind_seconds = t_blind.elapsed().as_secs_f64();

    // Pass 2: full green-aware anneal from the TreeSA start.
    let t_aware = Instant::now();
    let mut aware_tree = GreenTree::from_nested(
        &start,
        int_ixs.clone(),
        int_iy.clone(),
        log2_sizes.clone(),
        is_complex.to_vec(),
        config.merge_factor,
        config.ride_factor,
    );
    let green_blind_cost = aware_tree.total_cost();
    let full_anneal_cost = aware_tree.multi_anneal(
        &start_snap,
        config.nsteps,
        config.t0,
        config.t1,
        &config.seeds,
    );
    let (full_anneal_pass_volume, full_anneal_ride_volume, full_anneal_merge_volume) =
        aware_tree.breakdown();
    let aware_seconds = t_aware.elapsed().as_secs_f64();

    // Pass 3: convert the best real skeleton, then low-temperature polish.
    let t_polish = Instant::now();
    let mut polish_tree = GreenTree::from_nested(
        &start,
        int_ixs,
        int_iy,
        log2_sizes,
        is_complex.to_vec(),
        config.merge_factor,
        config.ride_factor,
    );
    polish_tree.restore(&blind_snap);
    let convert_only_cost = polish_tree.total_cost();
    let polished_cost = polish_tree.multi_anneal(
        &blind_snap,
        config.nsteps / 2,
        config.polish_t0,
        config.polish_t1,
        &config.seeds,
    );
    let polish_seconds = t_polish.elapsed().as_secs_f64();

    let best_blind = blind_tree.to_nested(&code.ixs, &code.iy, &labels);
    let full_anneal = aware_tree.to_nested(&code.ixs, &code.iy, &labels);
    let polished = polish_tree.to_nested(&code.ixs, &code.iy, &labels);

    Some(GreenPipeOutcome {
        start,
        best_blind,
        polished,
        full_anneal,
        green_blind_cost,
        best_real_cost,
        convert_only_cost,
        polished_cost,
        full_anneal_cost,
        start_volume,
        full_anneal_pass_volume,
        full_anneal_ride_volume,
        full_anneal_merge_volume,
        seconds: (init_seconds, blind_seconds, aware_seconds, polish_seconds),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contraction_complexity;
    use crate::greedy::GreedyMethod;
    use crate::optimize_code;

    /// Ring of `nt` tensors (tensor i carries labels {i, (i+1) mod nt})
    /// plus `chords` random extra labels shared by two distinct tensors.
    /// Every label appears exactly twice -> valid scalar-output network.
    fn ring_network(
        nt: usize,
        chords: usize,
        seed: u64,
    ) -> (Vec<Vec<usize>>, Vec<usize>, Vec<f64>) {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let mut ixs: Vec<Vec<usize>> = (0..nt).map(|i| vec![i, (i + 1) % nt]).collect();
        let mut next_label = nt;
        for _ in 0..chords {
            let a = rng.random_range(0..nt);
            let mut b = rng.random_range(0..nt);
            if a == b {
                b = (b + 1) % nt;
            }
            ixs[a].push(next_label);
            ixs[b].push(next_label);
            next_label += 1;
        }
        let nlabels = next_label;
        let log2_sizes = vec![1.0; nlabels];
        (ixs, vec![], log2_sizes)
    }

    /// Deterministic mixed complexity pattern (~40% complex).
    fn mixed_complex(nt: usize) -> Vec<bool> {
        (0..nt).map(|i| (i * 7 + 3) % 5 < 2).collect()
    }

    /// Naive open-label cost of node `i`, computed by walking the subtree
    /// leaf sets from scratch. Independent of the incremental machinery.
    fn naive_real_cost(tree: &GreenTree, i: usize) -> f64 {
        fn collect_counts(tree: &GreenTree, i: usize, acc: &mut HashMap<usize, i16>) {
            if tree.is_leaf(i) {
                for &l in &tree.ixs[tree.tensoridx[i]] {
                    *acc.entry(l).or_insert(0) += 1;
                }
            } else {
                collect_counts(tree, tree.left[i], acc);
                collect_counts(tree, tree.right[i], acc);
            }
        }
        let a = tree.left[i];
        let b = tree.right[i];
        let mut ca = HashMap::new();
        let mut cb = HashMap::new();
        collect_counts(tree, a, &mut ca);
        collect_counts(tree, b, &mut cb);
        let mut open = 0.0f64;
        for l in 0..tree.nlabels {
            let x = ca.get(&l).copied().unwrap_or(0);
            let y = cb.get(&l).copied().unwrap_or(0);
            let tot = tree.total[l];
            if (x > 0 && x < tot)
                || (y > 0 && y < tot)
                || (tree.is_output[l] && (x > 0 || y > 0))
            {
                open += tree.log2_sizes[l];
            }
        }
        open.exp2()
    }

    fn assert_tree_matches_naive(tree: &GreenTree) {
        for i in 0..tree.nnodes() {
            if tree.is_leaf(i) {
                continue;
            }
            assert_eq!(
                tree.realcost[i],
                naive_real_cost(tree, i),
                "realcost mismatch at node {i}"
            );
            let ga = tree.green[tree.left[i]];
            let gb = tree.green[tree.right[i]];
            let factor = if ga && gb {
                tree.merge_factor
            } else if ga || gb {
                tree.ride_factor
            } else {
                1.0
            };
            assert_eq!(
                tree.stepcost[i],
                tree.realcost[i] * factor,
                "stepcost mismatch at node {i}"
            );
        }
    }

    fn greedy_start(ixs: &[Vec<usize>], iy: &[usize], log2_sizes: &[f64]) -> NestedEinsum<usize> {
        let code = EinCode::new(ixs.to_vec(), iy.to_vec());
        let sizes: HashMap<usize, usize> = log2_sizes
            .iter()
            .enumerate()
            .map(|(l, s)| (l, 2usize.pow(*s as u32)))
            .collect();
        optimize_code(&code, &sizes, &GreedyMethod::default()).expect("greedy start")
    }

    fn make_tree(
        ixs: &[Vec<usize>],
        iy: &[usize],
        log2_sizes: &[f64],
        is_complex: &[bool],
        merge: f64,
        ride: f64,
    ) -> GreenTree {
        let start = greedy_start(ixs, iy, log2_sizes);
        GreenTree::from_nested(
            &start,
            ixs.to_vec(),
            iy.to_vec(),
            log2_sizes.to_vec(),
            is_complex.to_vec(),
            merge,
            ride,
        )
    }

    #[test]
    fn refresh_matches_naive_on_start_tree() {
        let (ixs, iy, log2_sizes) = ring_network(24, 6, 5);
        let is_complex = mixed_complex(ixs.len());
        let tree = make_tree(&ixs, &iy, &log2_sizes, &is_complex, 3.0, 2.0);
        assert_tree_matches_naive(&tree);
    }

    #[test]
    fn incremental_cost_matches_full_recompute_after_rotations() {
        let (ixs, iy, log2_sizes) = ring_network(24, 6, 11);
        let is_complex = mixed_complex(ixs.len());
        let mut tree = make_tree(&ixs, &iy, &log2_sizes, &is_complex, 3.0, 2.0);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(99);
        let mut scratch = RotationScratch::new(tree.nlabels);
        for step in 0..300 {
            if tree.try_rotation(&mut rng, &mut scratch).is_none() {
                continue;
            }
            // Accept every rotation. Now the incremental state must equal a
            // full recomputation from structure, bitwise.
            let mut check = tree.clone();
            check.recompute_from_structure();
            assert_eq!(tree.counts, check.counts, "counts drift at step {step}");
            assert_eq!(tree.green, check.green, "green drift at step {step}");
            assert_eq!(
                tree.realcost, check.realcost,
                "realcost drift at step {step}"
            );
            assert_eq!(
                tree.stepcost, check.stepcost,
                "stepcost drift at step {step}"
            );
            // The tree must also stay a valid binary tree covering every node.
            let mut seen = vec![false; tree.nnodes()];
            let mut stack = vec![tree.root];
            let mut visited = 0;
            while let Some(i) = stack.pop() {
                assert!(i < tree.nnodes() && !seen[i], "structure corrupt at step {step}");
                seen[i] = true;
                visited += 1;
                if tree.is_leaf(i) {
                    assert!(tree.tensoridx[i] != NONE, "leaf without tensor at step {step}");
                } else {
                    stack.push(tree.left[i]);
                    stack.push(tree.right[i]);
                }
            }
            assert_eq!(visited, tree.nnodes(), "tree fragmented at step {step}");
            if step % 50 == 0 {
                assert_tree_matches_naive(&tree);
            }
        }
    }

    #[test]
    fn undo_restores_exact_state() {
        let (ixs, iy, log2_sizes) = ring_network(16, 4, 17);
        let is_complex = mixed_complex(ixs.len());
        let mut tree = make_tree(&ixs, &iy, &log2_sizes, &is_complex, 3.0, 2.0);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        let mut scratch = RotationScratch::new(tree.nlabels);
        let before = tree.snapshot();
        let counts_before = tree.counts.clone();
        let green_before = tree.green.clone();
        let step_before = tree.stepcost.clone();
        let real_before = tree.realcost.clone();
        let mut rotated = 0;
        while rotated < 50 {
            let Some(rot) = tree.try_rotation(&mut rng, &mut scratch) else {
                continue;
            };
            tree.undo(&rot, &mut scratch);
            rotated += 1;
        }
        assert_eq!(tree.left, before.left);
        assert_eq!(tree.right, before.right);
        assert_eq!(tree.parent, before.parent);
        assert_eq!(tree.counts, counts_before);
        assert_eq!(tree.green, green_before);
        assert_eq!(tree.stepcost, step_before);
        assert_eq!(tree.realcost, real_before);
    }

    #[test]
    fn known_four_tensor_tree_costs() {
        // Ring of 4: ixs = [[0,1],[1,2],[2,3],[3,0]], complex [T,F,T,T],
        // tree ((0,1),(2,3)).
        let ixs = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]];
        let iy: Vec<usize> = vec![];
        let log2_sizes = vec![1.0; 4];
        let is_complex = vec![true, false, true, true];
        let t01 = NestedEinsum::<usize>::node(
            vec![NestedEinsum::leaf(0), NestedEinsum::leaf(1)],
            EinCode::new(vec![], vec![]),
        );
        let t23 = NestedEinsum::<usize>::node(
            vec![NestedEinsum::leaf(2), NestedEinsum::leaf(3)],
            EinCode::new(vec![], vec![]),
        );
        let root = NestedEinsum::node(vec![t01, t23], EinCode::new(vec![], vec![]));

        let tree = GreenTree::from_nested(
            &root,
            ixs,
            iy,
            log2_sizes,
            is_complex,
            3.0,
            2.0,
        );
        // Open sets use PER-CHILD counts (0 < count_in_child < total):
        // Node (0,1) contracts [0,1]x[1,2]: open {0,1,2} -> real 8, ride x2 -> 16.
        // Node (2,3) contracts [2,3]x[3,0]: open {0,2,3} -> real 8, merge x3 -> 24.
        // Root contracts [0,2]x[0,2]: open {0,2} -> real 4, merge x3 -> 12.
        assert_eq!(tree.total_real(), 20.0);
        assert_eq!(tree.total_cost(), 52.0);
        assert_eq!(tree.breakdown(), (0.0, 8.0, 12.0));

        // Green-blind factors make step cost == real cost everywhere.
        let blind = GreenTree::from_nested(
            &root,
            vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]],
            vec![],
            vec![1.0; 4],
            vec![true, false, true, true],
            1.0,
            1.0,
        );
        assert_eq!(blind.stepcost, blind.realcost);
        assert_eq!(blind.total_cost(), 20.0);
    }

    #[test]
    fn green_blind_volume_matches_contraction_complexity() {
        // With factors (1, 1) and unit log2 sizes, total_real must equal
        // 2^tc of the exported NestedEinsum (sum of per-step volumes).
        let (ixs, iy, log2_sizes) = ring_network(20, 5, 23);
        let is_complex = mixed_complex(ixs.len());
        let sizes: HashMap<usize, usize> = (0..log2_sizes.len()).map(|l| (l, 2)).collect();
        let code = EinCode::new(ixs.clone(), iy.clone());

        let mut tree = make_tree(&ixs, &iy, &log2_sizes, &is_complex, 1.0, 1.0);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(31);
        let mut scratch = RotationScratch::new(tree.nlabels);
        for _ in 0..200 {
            let Some(rot) = tree.try_rotation(&mut rng, &mut scratch) else {
                continue;
            };
            if rot.delta > 0.0 {
                tree.undo(&rot, &mut scratch);
            }
        }
        let labels: Vec<usize> = (0..log2_sizes.len()).collect();
        let nested = tree.to_nested(&code.ixs, &code.iy, &labels);
        let metrics = contraction_complexity(&nested, &sizes, &code.ixs);
        let volume_from_tc = 2f64.powf(metrics.tc);
        let rel = (volume_from_tc - tree.total_real()).abs() / tree.total_real();
        assert!(
            rel < 1e-9,
            "2^tc = {volume_from_tc}, total_real = {}",
            tree.total_real()
        );
    }

    #[test]
    fn caterpillar_anneal_strictly_improves() {
        // A left-linear tree on a chorded ring is far from optimal; the
        // (1, 1) anneal must strictly improve it (deterministic seeds).
        // (A plain ring is a bad instance: every contraction step there
        // opens exactly 3 labels, so the caterpillar is already optimal.)
        let (ixs, iy, log2_sizes) = ring_network(30, 8, 41);
        let is_complex = vec![false; ixs.len()];
        let mut start = NestedEinsum::<usize>::node(
            vec![NestedEinsum::leaf(0), NestedEinsum::leaf(1)],
            EinCode::new(vec![], vec![]),
        );
        for i in 2..ixs.len() {
            start = NestedEinsum::node(
                vec![start, NestedEinsum::leaf(i)],
                EinCode::new(vec![], vec![]),
            );
        }
        let mut tree = GreenTree::from_nested(
            &start,
            ixs,
            iy,
            log2_sizes,
            is_complex,
            1.0,
            1.0,
        );
        let initial = tree.total_cost();
        let snap = tree.snapshot();
        let best = tree.multi_anneal(&snap, 20_000, 1.0, 0.005, &[42, 7, 2026]);
        assert!(
            best < initial,
            "anneal should improve a caterpillar: {best} vs {initial}"
        );
        assert!(tree.total_cost() <= initial);
    }

    #[test]
    fn anneal_is_deterministic_per_seed() {
        let (ixs, iy, log2_sizes) = ring_network(20, 5, 43);
        let is_complex = mixed_complex(ixs.len());
        let run = || {
            let mut tree = make_tree(&ixs, &iy, &log2_sizes, &is_complex, 3.0, 2.0);
            let snap = tree.snapshot();
            tree.multi_anneal(&snap, 5_000, 1.0, 0.005, &[42, 7, 2026])
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn to_nested_produces_consistent_labels() {
        fn check<L: Label>(nested: &NestedEinsum<L>, original_ixs: &[Vec<L>], iy: &[L]) -> Vec<L> {
            match nested {
                NestedEinsum::Leaf { tensor_index } => original_ixs[*tensor_index].clone(),
                NestedEinsum::Node { args, eins } => {
                    assert_eq!(args.len(), 2);
                    assert_eq!(eins.ixs.len(), 2);
                    let lo = check(&args[0], original_ixs, iy);
                    let ro = check(&args[1], original_ixs, iy);
                    assert_eq!(lo, eins.ixs[0], "left child output mismatch");
                    assert_eq!(ro, eins.ixs[1], "right child output mismatch");
                    eins.iy.clone()
                }
            }
        }
        let (ixs, iy, log2_sizes) = ring_network(18, 4, 47);
        let is_complex = mixed_complex(ixs.len());
        let code = EinCode::new(ixs.clone(), iy.clone());
        let mut tree = make_tree(&ixs, &iy, &log2_sizes, &is_complex, 3.0, 2.0);
        let snap = tree.snapshot();
        tree.multi_anneal(&snap, 3_000, 1.0, 0.005, &[42]);
        let labels: Vec<usize> = (0..log2_sizes.len()).collect();
        let nested = tree.to_nested(&code.ixs, &code.iy, &labels);
        assert!(nested.is_binary());
        assert_eq!(nested.leaf_count(), ixs.len());
        let root_out = check(&nested, &code.ixs, &code.iy);
        assert_eq!(root_out, code.iy, "root output must match code.iy exactly");
    }

    #[test]
    fn optimize_with_green_end_to_end() {
        let (ixs, iy, log2_sizes) = ring_network(24, 6, 53);
        let is_complex = mixed_complex(ixs.len());
        let code = EinCode::new(ixs, iy);
        let sizes: HashMap<usize, usize> = (0..log2_sizes.len()).map(|l| (l, 2)).collect();
        let annealer = GreenAnnealer::green_aware()
            .with_nsteps(4_000)
            .with_initializer(crate::treesa::TreeSA::fast());
        let (nested, report) = annealer
            .optimize_with_green(&code, &sizes, &is_complex)
            .expect("green optimization");
        assert!(nested.is_binary());
        assert!(report.final_cost <= report.start_cost);
        let total = report.pass_volume + report.ride_volume + report.merge_volume;
        assert!(total > 0.0);
        let metrics = contraction_complexity(&nested, &sizes, &code.ixs);
        let rel = (2f64.powf(metrics.tc) - total).abs() / total;
        assert!(rel < 1e-9, "2^tc vs annealer volume: rel err {rel}");
    }

    #[test]
    fn green_pipeline_orders_passes() {
        let (ixs, iy, log2_sizes) = ring_network(24, 6, 59);
        let is_complex = mixed_complex(ixs.len());
        let code = EinCode::new(ixs, iy);
        let sizes: HashMap<usize, usize> = (0..log2_sizes.len()).map(|l| (l, 2)).collect();
        let config = GreenAnnealer::default()
            .with_nsteps(4_000)
            .with_initializer(crate::treesa::TreeSA::fast());
        let outcome = green_pipeline(&code, &sizes, &is_complex, &config)
            .expect("green pipeline");
        // Acceptance invariants: annealed <= start (same measure).
        assert!(outcome.best_real_cost <= outcome.start_volume);
        assert!(outcome.full_anneal_cost <= outcome.green_blind_cost);
        assert!(outcome.polished_cost <= outcome.convert_only_cost);
        // Volume bookkeeping of the full-anneal tree.
        let total = outcome.full_anneal_pass_volume
            + outcome.full_anneal_ride_volume
            + outcome.full_anneal_merge_volume;
        assert!(total > 0.0);
        // The exported trees are valid binary trees.
        assert!(outcome.best_blind.is_binary());
        assert!(outcome.polished.is_binary());
        assert!(outcome.full_anneal.is_binary());
    }

    #[test]
    fn output_labels_stay_open_when_inside_one_child() {
        // Non-scalar output: label 2 is in iy and occurs only in tensor 1.
        // It must remain open (and priced) at every ancestor step even
        // though 0 < count < total never holds for it.
        let ixs = vec![vec![0, 1], vec![1, 2]];
        let iy = vec![2];
        let log2_sizes = vec![1.0; 3];
        let is_complex = vec![false, false];
        let start = NestedEinsum::<usize>::node(
            vec![NestedEinsum::leaf(0), NestedEinsum::leaf(1)],
            EinCode::new(vec![], vec![]),
        );
        let tree = GreenTree::from_nested(
            &start,
            ixs,
            iy,
            log2_sizes,
            is_complex,
            1.0,
            1.0,
        );
        // Root step: open labels {1, 2} -> real cost 4.
        assert_eq!(tree.total_real(), 4.0);
        let labels: Vec<usize> = vec![0, 1, 2];
        let nested = tree.to_nested(&[vec![0usize, 1], vec![1, 2]], &[2], &labels);
        match &nested {
            NestedEinsum::Node { eins, .. } => assert_eq!(eins.iy, vec![2]),
            _ => panic!("expected internal root"),
        }
    }




}


