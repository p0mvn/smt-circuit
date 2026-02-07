//! Compressed Sparse Merkle Tree implementation.
//!
//! This module provides a memory-efficient alternative to [`SparseMerkleTree`]
//! that classifies subtrees by leaf count (Zero/Single/Double/Multi) instead
//! of materializing every internal node in a `BTreeMap`.
//!
//! For 51.7M leaves at height 53, this reduces memory from ~80 GB to ~6 GB
//! while producing identical [`Path`] and [`SparsePath`] proofs compatible
//! with the existing circuit chips (`PathChip`, `SparsePathChip`).
//!
//! Construction uses rayon for multi-core parallelism on large subtrees.

use crate::poseidon::FieldHasher;
use crate::smt::{gen_empty_hashes, Path, SparsePath, SparsePathEntry};
use anyhow::Result;
use ff::{FromUniformBytes, PrimeField};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Minimum number of leaves in a subtree to trigger parallel recursion via rayon.
const PARALLEL_THRESHOLD: usize = 1024;

// ============================================================
// Build progress tracking
// ============================================================

/// Tracks leaf processing progress during tree construction.
///
/// Uses an atomic counter so parallel rayon threads can safely
/// report progress. Logs at ~20 evenly-spaced intervals to avoid
/// flooding the output while still giving useful feedback.
struct BuildProgress {
    processed: AtomicUsize,
    total: usize,
    log_interval: usize,
}

impl BuildProgress {
    fn new(total: usize) -> Self {
        // Log roughly 20 times during the build, minimum interval of 1.
        let log_interval = (total / 20).max(1);
        BuildProgress {
            processed: AtomicUsize::new(0),
            total,
            log_interval,
        }
    }

    /// Report that `count` leaves have been processed.
    /// Logs progress when crossing an interval boundary or reaching completion.
    fn report(&self, count: usize) {
        if self.total == 0 {
            return;
        }
        let prev = self.processed.fetch_add(count, Ordering::Relaxed);
        let done = prev + count;
        // Log when we cross an interval boundary or reach the total.
        if prev / self.log_interval != done / self.log_interval || done >= self.total {
            let clamped = done.min(self.total);
            log::info!(
                "[CompressedSMT] Progress: {}/{} leaves ({:.1}%)",
                clamped,
                self.total,
                (clamped as f64 / self.total as f64) * 100.0
            );
        }
    }
}

// ============================================================
// CompressedNode enum
// ============================================================

/// Compressed node in the sparse Merkle tree.
///
/// Each variant represents a subtree classified by how many non-default
/// leaves it contains. Only `Multi` nodes allocate child pointers; the
/// other variants store O(1) data and compute their hashes eagerly
/// during construction.
enum CompressedNode<F: PrimeField> {
    /// Subtree has zero non-default leaves.
    /// Hash = `empty_hashes[level]` (looked up, not stored).
    Zero,

    /// Subtree has exactly one non-default leaf.
    /// Hash = leaf hashed up through empty siblings (computed during build).
    Single {
        leaf_index: u64,
        leaf_value: F,
        hash: F,
    },

    /// Subtree has exactly two non-default leaves.
    /// Hash = merge of two Single paths at their divergence point.
    Double {
        leaf_a: (u64, F),
        leaf_b: (u64, F),
        hash: F,
    },

    /// Subtree has 3+ non-default leaves.
    /// Hash = poseidon(left.hash, right.hash), stored explicitly.
    Multi {
        hash: F,
        left: Box<CompressedNode<F>>,
        right: Box<CompressedNode<F>>,
    },
}

impl<F: PrimeField> CompressedNode<F> {
    /// Returns the hash of this node at the given level.
    ///
    /// For `Zero` nodes, returns the precomputed empty hash at that level.
    /// For all other variants, returns the stored hash.
    fn hash_at_level(&self, level: usize, empty_hashes: &[F]) -> F {
        match self {
            CompressedNode::Zero => empty_hashes[level],
            CompressedNode::Single { hash, .. }
            | CompressedNode::Double { hash, .. }
            | CompressedNode::Multi { hash, .. } => *hash,
        }
    }

    /// Returns a string name for the variant (for testing/debugging).
    #[cfg(test)]
    fn variant_name(&self) -> &'static str {
        match self {
            CompressedNode::Zero => "Zero",
            CompressedNode::Single { .. } => "Single",
            CompressedNode::Double { .. } => "Double",
            CompressedNode::Multi { .. } => "Multi",
        }
    }
}

// ============================================================
// Hash helper functions
// ============================================================

/// Hash one leaf up through `level` layers of empty siblings.
///
/// Starting from `leaf_value`, at each layer `l` (0 ≤ l < level), the
/// current hash is combined with `empty_hashes[l]` on the side determined
/// by bit `l` of `leaf_index`.
fn compute_single_hash<F: PrimeField, H: FieldHasher<F, 2>>(
    (leaf_index, leaf_value): (u64, F),
    level: usize,
    hasher: &H,
    empty_hashes: &[F],
) -> F {
    let mut h = leaf_value;
    for l in 0..level {
        let bit = (leaf_index >> l) & 1;
        h = if bit == 0 {
            hasher.hash([h, empty_hashes[l]]).unwrap()
        } else {
            hasher.hash([empty_hashes[l], h]).unwrap()
        };
    }
    h
}

/// Hash two leaves by finding their divergence point and merging.
///
/// 1. Find the highest differing bit between the two indices (`div_level`).
/// 2. Hash each leaf independently up to `div_level - 1` levels.
/// 3. Merge the two hashes at the divergence level.
/// 4. Continue hashing up through empty siblings to `level`.
fn compute_double_hash<F: PrimeField, H: FieldHasher<F, 2>>(
    a: (u64, F),
    b: (u64, F),
    level: usize,
    hasher: &H,
    empty_hashes: &[F],
) -> F {
    let xor = a.0 ^ b.0;
    debug_assert!(xor != 0, "duplicate leaf indices in Double node");
    let div_level = (64 - xor.leading_zeros()) as usize;
    debug_assert!(
        div_level <= level,
        "divergence level {} exceeds node level {}",
        div_level,
        level
    );

    // Hash each leaf up to just below the divergence point
    let a_hash = compute_single_hash(a, div_level - 1, hasher, empty_hashes);
    let b_hash = compute_single_hash(b, div_level - 1, hasher, empty_hashes);

    // Merge at the divergence level
    let a_bit = (a.0 >> (div_level - 1)) & 1;
    let (left, right) = if a_bit == 0 {
        (a_hash, b_hash)
    } else {
        (b_hash, a_hash)
    };
    let mut h = hasher.hash([left, right]).unwrap();

    // Continue up from div_level to the node's level
    for l in div_level..level {
        let bit = (a.0 >> l) & 1; // a and b agree above divergence
        h = if bit == 0 {
            hasher.hash([h, empty_hashes[l]]).unwrap()
        } else {
            hasher.hash([empty_hashes[l], h]).unwrap()
        };
    }
    h
}

// ============================================================
// Top-down recursive build with rayon parallelism
// ============================================================

/// Recursively build a compressed subtree from sorted leaves.
///
/// `level` is the height of the current subtree (N at root, 0 at leaves).
/// Leaves must be sorted by index. Uses rayon::join for parallelism when
/// the leaf count exceeds `PARALLEL_THRESHOLD`.
fn build<F: PrimeField, H: FieldHasher<F, 2> + Sync>(
    leaves: &[(u64, F)],
    level: usize,
    hasher: &H,
    empty_hashes: &[F],
    progress: &BuildProgress,
) -> CompressedNode<F> {
    match leaves.len() {
        0 => CompressedNode::Zero,
        1 => {
            let hash = compute_single_hash(leaves[0], level, hasher, empty_hashes);
            progress.report(1);
            CompressedNode::Single {
                leaf_index: leaves[0].0,
                leaf_value: leaves[0].1,
                hash,
            }
        }
        2 => {
            let hash =
                compute_double_hash(leaves[0], leaves[1], level, hasher, empty_hashes);
            progress.report(2);
            CompressedNode::Double {
                leaf_a: leaves[0],
                leaf_b: leaves[1],
                hash,
            }
        }
        _ => {
            // Split by bit (level - 1) of leaf index:
            // left children have bit = 0, right children have bit = 1.
            let split =
                leaves.partition_point(|(idx, _)| (*idx >> (level - 1)) & 1 == 0);
            let (left_leaves, right_leaves) = leaves.split_at(split);

            // Parallel recursion for large subtrees
            let (left, right) = if leaves.len() > PARALLEL_THRESHOLD {
                rayon::join(
                    || build(left_leaves, level - 1, hasher, empty_hashes, progress),
                    || build(right_leaves, level - 1, hasher, empty_hashes, progress),
                )
            } else {
                (
                    build(left_leaves, level - 1, hasher, empty_hashes, progress),
                    build(right_leaves, level - 1, hasher, empty_hashes, progress),
                )
            };

            let left_hash = left.hash_at_level(level - 1, empty_hashes);
            let right_hash = right.hash_at_level(level - 1, empty_hashes);
            let hash = hasher.hash([left_hash, right_hash]).unwrap();

            CompressedNode::Multi {
                hash,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
    }
}

// ============================================================
// Dense proof helper
// ============================================================

/// Fill `path` and `direction_bits` arrays for a dense membership proof.
///
/// Traverses the compressed tree from the root toward the target leaf,
/// filling in the (left_hash, right_hash) pair and direction bit at each
/// of the N levels. For non-existent leaves the path is filled with
/// `empty_hashes[0]` as the leaf value and the correct sibling hashes
/// from the tree, matching [`SparseMerkleTree::generate_membership_proof`]
/// behavior.
fn fill_path<F: PrimeField, H: FieldHasher<F, 2>>(
    node: &CompressedNode<F>,
    target_index: u64,
    level: usize,
    hasher: &H,
    empty_hashes: &[F],
    path: &mut [(F, F)],
    direction_bits: &mut [bool],
) {
    match node {
        CompressedNode::Zero => {
            // Non-existent leaf in an all-empty subtree.
            // The leaf value is empty_hashes[0] (default leaf).
            let mut h = empty_hashes[0];
            for l in 0..level {
                let bit = (target_index >> l) & 1;
                let sibling = empty_hashes[l];
                if bit == 0 {
                    path[l] = (h, sibling);
                    direction_bits[l] = false;
                } else {
                    path[l] = (sibling, h);
                    direction_bits[l] = true;
                }
                h = hasher.hash([path[l].0, path[l].1]).unwrap();
            }
        }
        CompressedNode::Single {
            leaf_index,
            leaf_value,
            ..
        } => {
            if *leaf_index == target_index {
                // Existing leaf: hash up through empty siblings.
                let mut h = *leaf_value;
                for l in 0..level {
                    let bit = (target_index >> l) & 1;
                    let sibling = empty_hashes[l];
                    if bit == 0 {
                        path[l] = (h, sibling);
                        direction_bits[l] = false;
                    } else {
                        path[l] = (sibling, h);
                        direction_bits[l] = true;
                    }
                    h = hasher.hash([path[l].0, path[l].1]).unwrap();
                }
            } else {
                // Non-existent leaf within a Single subtree.
                // Target is empty; the existing leaf appears as a
                // sibling at the divergence level.
                let xor = target_index ^ *leaf_index;
                let div_level = (64 - xor.leading_zeros()) as usize;
                let mut h = empty_hashes[0];

                // Below divergence: empty siblings
                for l in 0..(div_level - 1) {
                    let bit = (target_index >> l) & 1;
                    let sibling = empty_hashes[l];
                    if bit == 0 {
                        path[l] = (h, sibling);
                        direction_bits[l] = false;
                    } else {
                        path[l] = (sibling, h);
                        direction_bits[l] = true;
                    }
                    h = hasher.hash([path[l].0, path[l].1]).unwrap();
                }

                // At divergence: existing leaf's hash is the sibling
                {
                    let l = div_level - 1;
                    let other_hash = compute_single_hash(
                        (*leaf_index, *leaf_value),
                        div_level - 1,
                        hasher,
                        empty_hashes,
                    );
                    let bit = (target_index >> l) & 1;
                    if bit == 0 {
                        path[l] = (h, other_hash);
                        direction_bits[l] = false;
                    } else {
                        path[l] = (other_hash, h);
                        direction_bits[l] = true;
                    }
                    h = hasher.hash([path[l].0, path[l].1]).unwrap();
                }

                // Above divergence: empty siblings
                for l in div_level..level {
                    let bit = (target_index >> l) & 1;
                    let sibling = empty_hashes[l];
                    if bit == 0 {
                        path[l] = (h, sibling);
                        direction_bits[l] = false;
                    } else {
                        path[l] = (sibling, h);
                        direction_bits[l] = true;
                    }
                    h = hasher.hash([path[l].0, path[l].1]).unwrap();
                }
            }
        }
        CompressedNode::Double { leaf_a, leaf_b, .. } => {
            let matched = if leaf_a.0 == target_index {
                Some((leaf_a, leaf_b))
            } else if leaf_b.0 == target_index {
                Some((leaf_b, leaf_a))
            } else {
                None
            };

            if let Some((target_leaf, other_leaf)) = matched {
                // Existing leaf in Double subtree (unchanged logic).
                let xor = target_leaf.0 ^ other_leaf.0;
                let div_level = (64 - xor.leading_zeros()) as usize;
                let mut h = target_leaf.1;

                // Below divergence: empty siblings
                for l in 0..(div_level - 1) {
                    let bit = (target_index >> l) & 1;
                    let sibling = empty_hashes[l];
                    if bit == 0 {
                        path[l] = (h, sibling);
                        direction_bits[l] = false;
                    } else {
                        path[l] = (sibling, h);
                        direction_bits[l] = true;
                    }
                    h = hasher.hash([path[l].0, path[l].1]).unwrap();
                }

                // At divergence level: other leaf's hash is the sibling
                {
                    let l = div_level - 1;
                    let other_hash =
                        compute_single_hash(*other_leaf, div_level - 1, hasher, empty_hashes);
                    let bit = (target_index >> l) & 1;
                    if bit == 0 {
                        path[l] = (h, other_hash);
                        direction_bits[l] = false;
                    } else {
                        path[l] = (other_hash, h);
                        direction_bits[l] = true;
                    }
                    h = hasher.hash([path[l].0, path[l].1]).unwrap();
                }

                // Above divergence: empty siblings
                for l in div_level..level {
                    let bit = (target_index >> l) & 1;
                    let sibling = empty_hashes[l];
                    if bit == 0 {
                        path[l] = (h, sibling);
                        direction_bits[l] = false;
                    } else {
                        path[l] = (sibling, h);
                        direction_bits[l] = true;
                    }
                    h = hasher.hash([path[l].0, path[l].1]).unwrap();
                }
            } else {
                // Non-existent target in a Double subtree.
                // Both existing leaves appear as siblings on the target's path.
                let xor_a = target_index ^ leaf_a.0;
                let xor_b = target_index ^ leaf_b.0;
                let div_a = (64 - xor_a.leading_zeros()) as usize;
                let div_b = (64 - xor_b.leading_zeros()) as usize;
                let mut h = empty_hashes[0];

                if div_a == div_b {
                    // Both leaves diverge from target at the same level.
                    // They appear together as a combined Double sibling.

                    // Below divergence: empty siblings
                    for l in 0..(div_a - 1) {
                        let bit = (target_index >> l) & 1;
                        let sibling = empty_hashes[l];
                        if bit == 0 {
                            path[l] = (h, sibling);
                            direction_bits[l] = false;
                        } else {
                            path[l] = (sibling, h);
                            direction_bits[l] = true;
                        }
                        h = hasher.hash([path[l].0, path[l].1]).unwrap();
                    }

                    // At divergence: combined hash of both leaves
                    {
                        let l = div_a - 1;
                        let combined = compute_double_hash(
                            *leaf_a, *leaf_b, div_a - 1, hasher, empty_hashes,
                        );
                        let bit = (target_index >> l) & 1;
                        if bit == 0 {
                            path[l] = (h, combined);
                            direction_bits[l] = false;
                        } else {
                            path[l] = (combined, h);
                            direction_bits[l] = true;
                        }
                        h = hasher.hash([path[l].0, path[l].1]).unwrap();
                    }

                    // Above divergence: empty siblings
                    for l in div_a..level {
                        let bit = (target_index >> l) & 1;
                        let sibling = empty_hashes[l];
                        if bit == 0 {
                            path[l] = (h, sibling);
                            direction_bits[l] = false;
                        } else {
                            path[l] = (sibling, h);
                            direction_bits[l] = true;
                        }
                        h = hasher.hash([path[l].0, path[l].1]).unwrap();
                    }
                } else {
                    // Leaves diverge from target at different levels.
                    // The closer leaf is a sibling at the lower level,
                    // the farther leaf at the higher level.
                    let (close, far, close_div, far_div) = if div_a < div_b {
                        (leaf_a, leaf_b, div_a, div_b)
                    } else {
                        (leaf_b, leaf_a, div_b, div_a)
                    };

                    // Below close divergence: empty siblings
                    for l in 0..(close_div - 1) {
                        let bit = (target_index >> l) & 1;
                        let sibling = empty_hashes[l];
                        if bit == 0 {
                            path[l] = (h, sibling);
                            direction_bits[l] = false;
                        } else {
                            path[l] = (sibling, h);
                            direction_bits[l] = true;
                        }
                        h = hasher.hash([path[l].0, path[l].1]).unwrap();
                    }

                    // At close divergence: close leaf as sibling
                    {
                        let l = close_div - 1;
                        let close_hash = compute_single_hash(
                            *close, close_div - 1, hasher, empty_hashes,
                        );
                        let bit = (target_index >> l) & 1;
                        if bit == 0 {
                            path[l] = (h, close_hash);
                            direction_bits[l] = false;
                        } else {
                            path[l] = (close_hash, h);
                            direction_bits[l] = true;
                        }
                        h = hasher.hash([path[l].0, path[l].1]).unwrap();
                    }

                    // Between close and far divergence: empty siblings
                    for l in close_div..(far_div - 1) {
                        let bit = (target_index >> l) & 1;
                        let sibling = empty_hashes[l];
                        if bit == 0 {
                            path[l] = (h, sibling);
                            direction_bits[l] = false;
                        } else {
                            path[l] = (sibling, h);
                            direction_bits[l] = true;
                        }
                        h = hasher.hash([path[l].0, path[l].1]).unwrap();
                    }

                    // At far divergence: far leaf as sibling
                    {
                        let l = far_div - 1;
                        let far_hash = compute_single_hash(
                            *far, far_div - 1, hasher, empty_hashes,
                        );
                        let bit = (target_index >> l) & 1;
                        if bit == 0 {
                            path[l] = (h, far_hash);
                            direction_bits[l] = false;
                        } else {
                            path[l] = (far_hash, h);
                            direction_bits[l] = true;
                        }
                        h = hasher.hash([path[l].0, path[l].1]).unwrap();
                    }

                    // Above far divergence: empty siblings
                    for l in far_div..level {
                        let bit = (target_index >> l) & 1;
                        let sibling = empty_hashes[l];
                        if bit == 0 {
                            path[l] = (h, sibling);
                            direction_bits[l] = false;
                        } else {
                            path[l] = (sibling, h);
                            direction_bits[l] = true;
                        }
                        h = hasher.hash([path[l].0, path[l].1]).unwrap();
                    }
                }
            }
        }
        CompressedNode::Multi { left, right, .. } => {
            let bit = (target_index >> (level - 1)) & 1;
            if bit == 0 {
                // Target is in left child
                fill_path(
                    left,
                    target_index,
                    level - 1,
                    hasher,
                    empty_hashes,
                    path,
                    direction_bits,
                );
                let left_hash = left.hash_at_level(level - 1, empty_hashes);
                let right_hash = right.hash_at_level(level - 1, empty_hashes);
                path[level - 1] = (left_hash, right_hash);
                direction_bits[level - 1] = false;
            } else {
                // Target is in right child
                fill_path(
                    right,
                    target_index,
                    level - 1,
                    hasher,
                    empty_hashes,
                    path,
                    direction_bits,
                );
                let left_hash = left.hash_at_level(level - 1, empty_hashes);
                let right_hash = right.hash_at_level(level - 1, empty_hashes);
                path[level - 1] = (left_hash, right_hash);
                direction_bits[level - 1] = true;
            }
        }
    }
}

// ============================================================
// Sparse proof helper
// ============================================================

/// Collect non-empty sibling entries for a sparse membership proof.
///
/// Traverses the compressed tree, adding a `SparsePathEntry` only when
/// the sibling hash differs from the precomputed empty hash at that level.
/// Entries are naturally produced in ascending level order (leaf -> root).
/// For non-existent leaves, entries are produced for the actual non-empty
/// siblings along the path, matching [`SparseMerkleTree`] behavior.
fn collect_sparse_entries<F: PrimeField, H: FieldHasher<F, 2>>(
    node: &CompressedNode<F>,
    target_index: u64,
    level: usize,
    hasher: &H,
    empty_hashes: &[F],
    entries: &mut Vec<SparsePathEntry<F>>,
) {
    match node {
        CompressedNode::Zero => {
            // Non-existent leaf in an all-empty subtree: no non-empty siblings.
        }
        CompressedNode::Single {
            leaf_index,
            leaf_value,
            ..
        } => {
            if *leaf_index == target_index {
                // All siblings within a Single node are empty hashes — no entries to add.
            } else {
                // Non-existent leaf: existing leaf is a non-empty sibling
                // at the divergence level.
                let xor = target_index ^ *leaf_index;
                let div_level = (64 - xor.leading_zeros()) as usize;
                let sibling_hash = compute_single_hash(
                    (*leaf_index, *leaf_value),
                    div_level - 1,
                    hasher,
                    empty_hashes,
                );
                let bit = (target_index >> (div_level - 1)) & 1;
                entries.push(SparsePathEntry {
                    sibling: sibling_hash,
                    direction_bit: bit == 1,
                    level: div_level - 1,
                });
            }
        }
        CompressedNode::Double { leaf_a, leaf_b, .. } => {
            if leaf_a.0 == target_index || leaf_b.0 == target_index {
                // Target matches one leaf; the other is a sibling at divergence.
                let other = if leaf_a.0 == target_index {
                    leaf_b
                } else {
                    leaf_a
                };
                let xor = target_index ^ other.0;
                let div_level = (64 - xor.leading_zeros()) as usize;
                let sibling_hash =
                    compute_single_hash(*other, div_level - 1, hasher, empty_hashes);
                let bit = (target_index >> (div_level - 1)) & 1;
                entries.push(SparsePathEntry {
                    sibling: sibling_hash,
                    direction_bit: bit == 1,
                    level: div_level - 1,
                });
            } else {
                // Non-existent target: both leaves appear as siblings.
                let xor_a = target_index ^ leaf_a.0;
                let xor_b = target_index ^ leaf_b.0;
                let div_a = (64 - xor_a.leading_zeros()) as usize;
                let div_b = (64 - xor_b.leading_zeros()) as usize;

                if div_a == div_b {
                    // Both leaves diverge at the same level — combined sibling.
                    let combined = compute_double_hash(
                        *leaf_a, *leaf_b, div_a - 1, hasher, empty_hashes,
                    );
                    let bit = (target_index >> (div_a - 1)) & 1;
                    entries.push(SparsePathEntry {
                        sibling: combined,
                        direction_bit: bit == 1,
                        level: div_a - 1,
                    });
                } else {
                    // Leaves at different divergence levels — two entries
                    // pushed in ascending level order.
                    let (close, far, close_div, far_div) = if div_a < div_b {
                        (leaf_a, leaf_b, div_a, div_b)
                    } else {
                        (leaf_b, leaf_a, div_b, div_a)
                    };
                    let close_hash = compute_single_hash(
                        *close, close_div - 1, hasher, empty_hashes,
                    );
                    let close_bit = (target_index >> (close_div - 1)) & 1;
                    entries.push(SparsePathEntry {
                        sibling: close_hash,
                        direction_bit: close_bit == 1,
                        level: close_div - 1,
                    });
                    let far_hash = compute_single_hash(
                        *far, far_div - 1, hasher, empty_hashes,
                    );
                    let far_bit = (target_index >> (far_div - 1)) & 1;
                    entries.push(SparsePathEntry {
                        sibling: far_hash,
                        direction_bit: far_bit == 1,
                        level: far_div - 1,
                    });
                }
            }
        }
        CompressedNode::Multi { left, right, .. } => {
            let bit = (target_index >> (level - 1)) & 1;
            let (target_child, sibling_child) = if bit == 0 {
                (left.as_ref(), right.as_ref())
            } else {
                (right.as_ref(), left.as_ref())
            };

            // Recurse into target child first (produces lower-level entries)
            collect_sparse_entries(
                target_child,
                target_index,
                level - 1,
                hasher,
                empty_hashes,
                entries,
            );

            // Add sibling if non-empty
            let sibling_hash = sibling_child.hash_at_level(level - 1, empty_hashes);
            if sibling_hash != empty_hashes[level - 1] {
                entries.push(SparsePathEntry {
                    sibling: sibling_hash,
                    direction_bit: bit == 1,
                    level: level - 1,
                });
            }
        }
    }
}

// ============================================================
// CompressedSMT public struct and API
// ============================================================

/// Compressed Sparse Merkle Tree.
///
/// A memory-efficient alternative to [`SparseMerkleTree`] that classifies
/// subtrees by their non-default leaf count instead of storing every
/// internal node. Produces identical [`Path`] and [`SparsePath`] proofs
/// compatible with the existing circuit chips.
///
/// # Type Parameters
///
/// * `F` — prime field element type
/// * `H` — field hasher (e.g. Poseidon)
/// * `N` — tree height (number of hash levels)
pub struct CompressedSMT<
    F: PrimeField + FromUniformBytes<64>,
    H: FieldHasher<F, 2>,
    const N: usize,
> {
    root: CompressedNode<F>,
    empty_hashes: [F; N],
    marker: PhantomData<H>,
}

impl<F: PrimeField + FromUniformBytes<64>, H: FieldHasher<F, 2>, const N: usize>
    CompressedSMT<F, H, N>
{
    /// Build a compressed SMT from leaves with u64 indices.
    ///
    /// Accepts the full 2^N address space. The tree is built top-down
    /// with rayon parallelism for large subtrees.
    ///
    /// # Arguments
    ///
    /// * `leaves` — map of (leaf_index → leaf_value), must have indices < 2^N
    /// * `hasher` — the field hasher (must be `Sync` for parallel construction)
    /// * `empty_leaf` — 64-byte seed for the default empty leaf value
    pub fn new(
        leaves: &BTreeMap<u64, F>,
        hasher: &H,
        empty_leaf: &[u8; 64],
    ) -> Result<Self>
    where
        H: Sync,
    {
        log::info!(
            "[CompressedSMT] Building tree: height={}, leaves={}",
            N,
            leaves.len()
        );
        let start = Instant::now();

        let empty_hashes = gen_empty_hashes::<F, H, N>(hasher, empty_leaf)?;
        log::debug!(
            "[CompressedSMT] Empty hashes computed in {:?}",
            start.elapsed()
        );

        let sorted_leaves: Vec<(u64, F)> =
            leaves.iter().map(|(&k, &v)| (k, v)).collect();

        // Validate indices are within the tree's address space
        if N < 64 {
            if let Some(&(max_idx, _)) = sorted_leaves.last() {
                assert!(
                    max_idx < (1u64 << N),
                    "leaf index {} exceeds tree capacity 2^{}",
                    max_idx,
                    N
                );
            }
        }

        let build_start = Instant::now();
        let progress = BuildProgress::new(sorted_leaves.len());
        let root = build(&sorted_leaves, N, hasher, &empty_hashes, &progress);

        log::info!(
            "[CompressedSMT] Tree built in {:?} (height={}, leaves={})",
            build_start.elapsed(),
            N,
            leaves.len()
        );
        log::info!(
            "[CompressedSMT] Total construction time: {:?}",
            start.elapsed()
        );

        Ok(CompressedSMT {
            root,
            empty_hashes,
            marker: PhantomData,
        })
    }

    /// Build from u32-indexed leaves (backward compatibility with `SparseMerkleTree::new`).
    pub fn new_from_u32(
        leaves: &BTreeMap<u32, F>,
        hasher: &H,
        empty_leaf: &[u8; 64],
    ) -> Result<Self>
    where
        H: Sync,
    {
        let leaves64: BTreeMap<u64, F> =
            leaves.iter().map(|(&k, &v)| (k as u64, v)).collect();
        Self::new(&leaves64, hasher, empty_leaf)
    }

    /// Returns the Merkle tree root hash.
    ///
    /// For an empty tree, returns `empty_hashes[N-1]` to match
    /// `SparseMerkleTree::root()` behavior.
    pub fn root(&self) -> F {
        match &self.root {
            CompressedNode::Zero => *self.empty_hashes.last().unwrap(),
            CompressedNode::Single { hash, .. }
            | CompressedNode::Double { hash, .. }
            | CompressedNode::Multi { hash, .. } => *hash,
        }
    }

    /// Generate a dense membership proof (all N levels).
    ///
    /// Returns a [`Path`] compatible with `PathChip`.
    ///
    /// For non-existent indices the returned path proves that the leaf
    /// value at `index` is the default empty leaf (`empty_hashes[0]`),
    /// with the correct sibling hashes from the tree. This matches the
    /// behavior of [`SparseMerkleTree::generate_membership_proof`] and
    /// is required for exclusion proofs.
    pub fn generate_membership_proof(&self, index: u64) -> Path<F, H, N> {
        let hasher = H::hasher();
        let mut path = [(F::ZERO, F::ZERO); N];
        let mut direction_bits = [false; N];

        fill_path(
            &self.root,
            index,
            N,
            &hasher,
            &self.empty_hashes,
            &mut path,
            &mut direction_bits,
        );

        Path {
            path,
            direction_bits,
            marker: PhantomData,
        }
    }

    /// Generate a sparse membership proof (only non-empty sibling levels).
    ///
    /// Returns a [`SparsePath`] compatible with `SparsePathChip`.
    /// The proof has K entries where K << N for sparse trees.
    ///
    /// For non-existent indices, the proof contains entries for the
    /// actual non-empty siblings along the path, matching
    /// [`SparseMerkleTree::generate_sparse_membership_proof`] behavior.
    pub fn generate_sparse_membership_proof(&self, index: u64) -> SparsePath<F, H> {
        let hasher = H::hasher();
        let mut entries = Vec::new();

        collect_sparse_entries(
            &self.root,
            index,
            N,
            &hasher,
            &self.empty_hashes,
            &mut entries,
        );

        // Entries are already in ascending level order due to recursion-first traversal.
        SparsePath {
            entries,
            tree_height: N,
            empty_hashes: self.empty_hashes.to_vec(),
            marker: PhantomData,
        }
    }

    /// Access the root node (for testing/inspection).
    #[cfg(test)]
    fn root_node(&self) -> &CompressedNode<F> {
        &self.root
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poseidon2::Poseidon2;
    use crate::smt::SparseMerkleTree;
    use ff::Field;
    use pasta_curves::Fp;
    use rand::rngs::OsRng;
    use rand::Rng;
    use std::collections::BTreeMap;

    type TestHasher = Poseidon2<Fp, 2>;

    /// Helper: create both the original SMT and compressed SMT from the same leaves.
    fn create_both<const N: usize>(
        leaves: &BTreeMap<u32, Fp>,
    ) -> (
        SparseMerkleTree<Fp, TestHasher, N>,
        CompressedSMT<Fp, TestHasher, N>,
    ) {
        let hasher = Poseidon2::<Fp, 2>::new();
        let empty_leaf = [0u8; 64];
        let smt = SparseMerkleTree::new(leaves, &hasher, &empty_leaf).unwrap();
        let csmt = CompressedSMT::new_from_u32(leaves, &hasher, &empty_leaf).unwrap();
        (smt, csmt)
    }

    // ---- Root correctness tests ----

    #[test]
    fn test_root_matches_height_3() {
        let rng = OsRng;
        let leaves: BTreeMap<u32, Fp> =
            (0..3).map(|i| (i, Fp::random(rng))).collect();
        let (smt, csmt) = create_both::<3>(&leaves);
        assert_eq!(smt.root(), csmt.root());
    }

    #[test]
    fn test_root_matches_height_10() {
        let rng = OsRng;
        let leaves: BTreeMap<u32, Fp> =
            (0..50).map(|i| (i, Fp::random(rng))).collect();
        let (smt, csmt) = create_both::<10>(&leaves);
        assert_eq!(smt.root(), csmt.root());
    }

    #[test]
    fn test_root_matches_sparse_leaves() {
        let rng = OsRng;
        let indices = [0u32, 5, 13, 100, 500, 1000];
        let leaves: BTreeMap<u32, Fp> =
            indices.iter().map(|&i| (i, Fp::random(rng))).collect();
        let (smt, csmt) = create_both::<20>(&leaves);
        assert_eq!(smt.root(), csmt.root());
    }

    #[test]
    fn test_root_single_leaf() {
        let rng = OsRng;
        let leaves: BTreeMap<u32, Fp> = [(0, Fp::random(rng))].into_iter().collect();
        let (smt, csmt) = create_both::<10>(&leaves);
        assert_eq!(smt.root(), csmt.root());
    }

    #[test]
    fn test_root_two_leaves() {
        let rng = OsRng;
        let leaves: BTreeMap<u32, Fp> = [(0, Fp::random(rng)), (1, Fp::random(rng))]
            .into_iter()
            .collect();
        let (smt, csmt) = create_both::<10>(&leaves);
        assert_eq!(smt.root(), csmt.root());
    }

    #[test]
    fn test_root_two_leaves_wide_separation() {
        let rng = OsRng;
        let leaves: BTreeMap<u32, Fp> =
            [(0, Fp::random(rng)), (1023, Fp::random(rng))]
                .into_iter()
                .collect();
        let (smt, csmt) = create_both::<10>(&leaves);
        assert_eq!(smt.root(), csmt.root());
    }

    #[test]
    fn test_root_empty_tree() {
        let leaves: BTreeMap<u32, Fp> = BTreeMap::new();
        let (smt, csmt) = create_both::<10>(&leaves);
        assert_eq!(smt.root(), csmt.root());
    }

    // ---- Node classification tests ----

    #[test]
    fn test_node_classification() {
        let hasher = Poseidon2::<Fp, 2>::new();
        let empty_leaf = [0u8; 64];

        // 0 leaves → Zero
        let leaves0: BTreeMap<u64, Fp> = BTreeMap::new();
        let csmt0 =
            CompressedSMT::<Fp, TestHasher, 5>::new(&leaves0, &hasher, &empty_leaf)
                .unwrap();
        assert_eq!(csmt0.root_node().variant_name(), "Zero");

        // 1 leaf → Single
        let leaves1: BTreeMap<u64, Fp> = [(0, Fp::from(42))].into_iter().collect();
        let csmt1 =
            CompressedSMT::<Fp, TestHasher, 5>::new(&leaves1, &hasher, &empty_leaf)
                .unwrap();
        assert_eq!(csmt1.root_node().variant_name(), "Single");

        // 2 leaves → Double
        let leaves2: BTreeMap<u64, Fp> =
            [(0, Fp::from(42)), (1, Fp::from(43))].into_iter().collect();
        let csmt2 =
            CompressedSMT::<Fp, TestHasher, 5>::new(&leaves2, &hasher, &empty_leaf)
                .unwrap();
        assert_eq!(csmt2.root_node().variant_name(), "Double");

        // 3 leaves → Multi
        let leaves3: BTreeMap<u64, Fp> = [
            (0, Fp::from(42)),
            (1, Fp::from(43)),
            (2, Fp::from(44)),
        ]
        .into_iter()
        .collect();
        let csmt3 =
            CompressedSMT::<Fp, TestHasher, 5>::new(&leaves3, &hasher, &empty_leaf)
                .unwrap();
        assert_eq!(csmt3.root_node().variant_name(), "Multi");
    }

    // ---- Dense proof tests ----

    #[test]
    fn test_dense_proof_matches_original() {
        let rng = OsRng;
        let leaves: BTreeMap<u32, Fp> =
            (0..5).map(|i| (i, Fp::random(rng))).collect();
        let (smt, csmt) = create_both::<10>(&leaves);

        for &idx in leaves.keys() {
            let smt_proof = smt.generate_membership_proof(idx as u64);
            let csmt_proof = csmt.generate_membership_proof(idx as u64);

            for level in 0..10 {
                assert_eq!(
                    smt_proof.path[level], csmt_proof.path[level],
                    "path mismatch at level {} for leaf {}",
                    level, idx
                );
                assert_eq!(
                    smt_proof.direction_bits[level], csmt_proof.direction_bits[level],
                    "direction_bit mismatch at level {} for leaf {}",
                    level, idx
                );
            }
        }
    }

    #[test]
    fn test_dense_proof_membership_check() {
        let poseidon = Poseidon2::<Fp, 2>::new();
        let rng = OsRng;
        let leaves: BTreeMap<u32, Fp> =
            (0..8).map(|i| (i, Fp::random(rng))).collect();
        let (_, csmt) = create_both::<10>(&leaves);

        for (&idx, &val) in &leaves {
            let proof = csmt.generate_membership_proof(idx as u64);
            let ok = proof
                .check_membership(&csmt.root(), &val, &poseidon)
                .unwrap();
            assert!(ok, "membership check failed for leaf {}", idx);
        }
    }

    #[test]
    fn test_dense_proof_sparse_leaves() {
        let rng = OsRng;
        let indices = [0u32, 5, 13, 100, 500, 1000];
        let leaves: BTreeMap<u32, Fp> =
            indices.iter().map(|&i| (i, Fp::random(rng))).collect();
        let (smt, csmt) = create_both::<20>(&leaves);
        let poseidon = Poseidon2::<Fp, 2>::new();

        for &idx in &indices {
            let smt_proof = smt.generate_membership_proof(idx as u64);
            let csmt_proof = csmt.generate_membership_proof(idx as u64);

            for level in 0..20 {
                assert_eq!(
                    smt_proof.path[level], csmt_proof.path[level],
                    "path mismatch at level {} for leaf {}",
                    level, idx
                );
            }

            let ok = csmt_proof
                .check_membership(&csmt.root(), &leaves[&idx], &poseidon)
                .unwrap();
            assert!(ok, "membership check failed for leaf {}", idx);
        }
    }

    // ---- Sparse proof tests ----

    #[test]
    fn test_sparse_proof_matches_original() {
        let rng = OsRng;
        let leaves: BTreeMap<u32, Fp> =
            (0..5).map(|i| (i, Fp::random(rng))).collect();
        let (smt, csmt) = create_both::<20>(&leaves);

        for &idx in leaves.keys() {
            let smt_sparse = smt.generate_sparse_membership_proof(idx as u64);
            let csmt_sparse = csmt.generate_sparse_membership_proof(idx as u64);

            assert_eq!(
                smt_sparse.entries.len(),
                csmt_sparse.entries.len(),
                "entry count mismatch for leaf {}",
                idx
            );

            for (i, (s, c)) in smt_sparse
                .entries
                .iter()
                .zip(csmt_sparse.entries.iter())
                .enumerate()
            {
                assert_eq!(
                    s.sibling, c.sibling,
                    "sibling mismatch at entry {} for leaf {}",
                    i, idx
                );
                assert_eq!(
                    s.direction_bit, c.direction_bit,
                    "direction_bit mismatch at entry {} for leaf {}",
                    i, idx
                );
                assert_eq!(
                    s.level, c.level,
                    "level mismatch at entry {} for leaf {}",
                    i, idx
                );
            }
        }
    }

    #[test]
    fn test_sparse_proof_full_root() {
        let poseidon = Poseidon2::<Fp, 2>::new();
        let rng = OsRng;
        let leaves: BTreeMap<u32, Fp> =
            (0..8).map(|i| (i, Fp::random(rng))).collect();
        let (_, csmt) = create_both::<20>(&leaves);

        for (&idx, &val) in &leaves {
            let sparse = csmt.generate_sparse_membership_proof(idx as u64);
            let full_root = sparse
                .calculate_full_root(&val, &poseidon, idx as u64)
                .unwrap();
            assert_eq!(
                full_root,
                csmt.root(),
                "full root mismatch for leaf {}",
                idx
            );
        }
    }

    #[test]
    fn test_sparse_proof_compact_root() {
        let poseidon = Poseidon2::<Fp, 2>::new();
        let rng = OsRng;
        let leaves: BTreeMap<u32, Fp> =
            (0..8).map(|i| (i, Fp::random(rng))).collect();
        let (smt, csmt) = create_both::<20>(&leaves);

        for (&idx, &val) in &leaves {
            let smt_sparse = smt.generate_sparse_membership_proof(idx as u64);
            let csmt_sparse = csmt.generate_sparse_membership_proof(idx as u64);

            let smt_compact = smt_sparse
                .calculate_compact_root(&val, &poseidon)
                .unwrap();
            let csmt_compact = csmt_sparse
                .calculate_compact_root(&val, &poseidon)
                .unwrap();

            assert_eq!(
                smt_compact, csmt_compact,
                "compact root mismatch for leaf {}",
                idx
            );
        }
    }

    // ---- Scale tests ----

    #[test]
    fn test_scale_1k_leaves() {
        let poseidon = Poseidon2::<Fp, 2>::new();
        let rng = OsRng;
        let leaves: BTreeMap<u32, Fp> =
            (0..1000).map(|i| (i, Fp::random(rng))).collect();
        let (smt, csmt) = create_both::<20>(&leaves);

        assert_eq!(smt.root(), csmt.root(), "roots should match for 1000 leaves");

        // Check a sample of proofs
        for idx in [0u32, 42, 500, 999] {
            let proof = csmt.generate_membership_proof(idx as u64);
            let ok = proof
                .check_membership(&csmt.root(), &leaves[&idx], &poseidon)
                .unwrap();
            assert!(ok, "membership check failed for leaf {}", idx);

            let sparse = csmt.generate_sparse_membership_proof(idx as u64);
            let full_root = sparse
                .calculate_full_root(&leaves[&idx], &poseidon, idx as u64)
                .unwrap();
            assert_eq!(
                full_root,
                csmt.root(),
                "sparse full root mismatch for leaf {}",
                idx
            );
        }
    }

    #[test]
    fn test_scale_height_53() {
        let mut rng = OsRng;
        let poseidon = Poseidon2::<Fp, 2>::new();
        let empty_leaf = [0u8; 64];

        // Create 10,000 leaves with random u64 indices in the 2^53 space
        let mut leaves: BTreeMap<u64, Fp> = BTreeMap::new();
        for _ in 0..10_000 {
            let idx: u64 = rng.gen_range(0..(1u64 << 53));
            let val = Fp::random(&mut rng);
            leaves.insert(idx, val);
        }

        let csmt = CompressedSMT::<Fp, TestHasher, 53>::new(
            &leaves,
            &poseidon,
            &empty_leaf,
        )
        .unwrap();

        // Root should be non-zero
        assert_ne!(csmt.root(), Fp::ZERO);

        // Sample a few leaves and verify proofs
        let sample_indices: Vec<u64> = leaves.keys().take(5).copied().collect();
        for &idx in &sample_indices {
            let proof = csmt.generate_membership_proof(idx);
            let ok = proof
                .check_membership(&csmt.root(), &leaves[&idx], &poseidon)
                .unwrap();
            assert!(ok, "membership check failed for leaf {}", idx);

            let sparse = csmt.generate_sparse_membership_proof(idx);
            let full_root = sparse
                .calculate_full_root(&leaves[&idx], &poseidon, idx)
                .unwrap();
            assert_eq!(
                full_root,
                csmt.root(),
                "sparse full root mismatch for leaf {}",
                idx
            );
        }
    }

    // ---- Non-existent index (exclusion) proof tests ----

    #[test]
    fn test_nonexistent_proof_zero_node() {
        // Empty tree (Zero root): proofs for any index should match
        // between CompressedSMT and SparseMerkleTree.
        //
        // Note: check_membership is not used here because the library's
        // root() for an empty tree returns empty_hashes[N-1], while
        // calculate_root hashes through all N levels producing a
        // different value. This is a pre-existing library convention
        // and doesn't affect real use (exclusion proofs require at
        // least one leaf in the tree).
        let leaves: BTreeMap<u32, Fp> = BTreeMap::new();
        let (smt, csmt) = create_both::<10>(&leaves);

        // Roots should match between the two implementations
        assert_eq!(smt.root(), csmt.root());

        for idx in [0u64, 1, 512, 1023] {
            let csmt_proof = csmt.generate_membership_proof(idx);
            let smt_proof = smt.generate_membership_proof(idx);

            // Path should match the original SMT field-by-field
            for level in 0..10 {
                assert_eq!(
                    smt_proof.path[level], csmt_proof.path[level],
                    "path mismatch at level {} for non-existent index {}",
                    level, idx
                );
                assert_eq!(
                    smt_proof.direction_bits[level], csmt_proof.direction_bits[level],
                    "direction_bit mismatch at level {} for non-existent index {}",
                    level, idx
                );
            }
        }

        // Also verify sparse proofs match
        for idx in [0u64, 1, 512, 1023] {
            let csmt_sparse = csmt.generate_sparse_membership_proof(idx);
            let smt_sparse = smt.generate_sparse_membership_proof(idx);
            assert_eq!(
                smt_sparse.entries.len(),
                csmt_sparse.entries.len(),
                "sparse entry count mismatch for index {} in empty tree",
                idx
            );
        }
    }

    #[test]
    fn test_nonexistent_proof_single_mismatch() {
        // Tree with one leaf at index 5: proof for index 3 should
        // return the empty leaf with the correct sibling hashes.
        let poseidon = Poseidon2::<Fp, 2>::new();
        let empty_leaf = [0u8; 64];
        let rng = OsRng;
        let leaves: BTreeMap<u32, Fp> = [(5, Fp::random(rng))].into_iter().collect();
        let (smt, csmt) = create_both::<10>(&leaves);
        let empty_val = Fp::from_uniform_bytes(&empty_leaf);

        for idx in [0u64, 3, 4, 6, 100, 1023] {
            let csmt_proof = csmt.generate_membership_proof(idx);
            let smt_proof = smt.generate_membership_proof(idx);

            for level in 0..10 {
                assert_eq!(
                    smt_proof.path[level], csmt_proof.path[level],
                    "path mismatch at level {} for non-existent index {} (single leaf at 5)",
                    level, idx
                );
                assert_eq!(
                    smt_proof.direction_bits[level], csmt_proof.direction_bits[level],
                    "direction_bit mismatch at level {} for non-existent index {}",
                    level, idx
                );
            }

            let ok = csmt_proof
                .check_membership(&csmt.root(), &empty_val, &poseidon)
                .unwrap();
            assert!(ok, "membership check failed for non-existent index {}", idx);
        }
    }

    #[test]
    fn test_nonexistent_proof_double_mismatch() {
        // Tree with leaves at indices 2 and 7: proof for index 4
        // should work correctly.
        let poseidon = Poseidon2::<Fp, 2>::new();
        let empty_leaf = [0u8; 64];
        let rng = OsRng;
        let leaves: BTreeMap<u32, Fp> = [(2, Fp::random(rng)), (7, Fp::random(rng))]
            .into_iter()
            .collect();
        let (smt, csmt) = create_both::<10>(&leaves);
        let empty_val = Fp::from_uniform_bytes(&empty_leaf);

        // Test indices that exercise different divergence patterns:
        // - idx 0: diverges from leaf 2 at level 2, leaf 7 at level 3
        // - idx 4: diverges from both at level 3 (same-level divergence)
        // - idx 3: diverges from leaf 2 at level 1, leaf 7 at level 3
        // - idx 1023: diverges at high level
        for idx in [0u64, 1, 3, 4, 5, 6, 8, 100, 1023] {
            let csmt_proof = csmt.generate_membership_proof(idx);
            let smt_proof = smt.generate_membership_proof(idx);

            for level in 0..10 {
                assert_eq!(
                    smt_proof.path[level], csmt_proof.path[level],
                    "path mismatch at level {} for non-existent index {} (leaves at 2,7)",
                    level, idx
                );
                assert_eq!(
                    smt_proof.direction_bits[level], csmt_proof.direction_bits[level],
                    "direction_bit mismatch at level {} for non-existent index {}",
                    level, idx
                );
            }

            let ok = csmt_proof
                .check_membership(&csmt.root(), &empty_val, &poseidon)
                .unwrap();
            assert!(ok, "membership check failed for non-existent index {}", idx);
        }
    }

    #[test]
    fn test_nonexistent_proof_multi_descend_into_zero() {
        // Tree with leaves clustered on the left side (indices 0..8),
        // probing an index on the empty right side.
        let poseidon = Poseidon2::<Fp, 2>::new();
        let empty_leaf = [0u8; 64];
        let rng = OsRng;
        let leaves: BTreeMap<u32, Fp> =
            (0..8).map(|i| (i, Fp::random(rng))).collect();
        let (smt, csmt) = create_both::<10>(&leaves);
        let empty_val = Fp::from_uniform_bytes(&empty_leaf);

        // These indices are in the empty right half of the tree
        for idx in [512u64, 600, 1023] {
            let csmt_proof = csmt.generate_membership_proof(idx);
            let smt_proof = smt.generate_membership_proof(idx);

            for level in 0..10 {
                assert_eq!(
                    smt_proof.path[level], csmt_proof.path[level],
                    "path mismatch at level {} for non-existent index {} (multi→zero)",
                    level, idx
                );
                assert_eq!(
                    smt_proof.direction_bits[level], csmt_proof.direction_bits[level],
                    "direction_bit mismatch at level {} for non-existent index {}",
                    level, idx
                );
            }

            let ok = csmt_proof
                .check_membership(&csmt.root(), &empty_val, &poseidon)
                .unwrap();
            assert!(ok, "membership check failed for non-existent index {}", idx);
        }

        // Also probe non-existent indices within the populated half
        for idx in [9u64, 15, 100] {
            let csmt_proof = csmt.generate_membership_proof(idx);
            let smt_proof = smt.generate_membership_proof(idx);

            for level in 0..10 {
                assert_eq!(
                    smt_proof.path[level], csmt_proof.path[level],
                    "path mismatch at level {} for non-existent index {} (within multi)",
                    level, idx
                );
            }

            let ok = csmt_proof
                .check_membership(&csmt.root(), &empty_val, &poseidon)
                .unwrap();
            assert!(ok, "membership check failed for non-existent index {}", idx);
        }
    }

    #[test]
    fn test_nonexistent_proof_matches_original_smt() {
        // For several tree configurations, insert random leaves then
        // probe non-existent indices — assert CompressedSMT matches
        // SparseMerkleTree field-by-field.
        let rng = OsRng;
        let empty_leaf = [0u8; 64];
        let poseidon = Poseidon2::<Fp, 2>::new();
        let empty_val = Fp::from_uniform_bytes(&empty_leaf);

        // Height 3, 3 leaves
        {
            let leaves: BTreeMap<u32, Fp> =
                [(0, Fp::random(rng)), (3, Fp::random(rng)), (7, Fp::random(rng))]
                    .into_iter()
                    .collect();
            let (smt, csmt) = create_both::<3>(&leaves);
            for idx in [1u64, 2, 4, 5, 6] {
                let smt_proof = smt.generate_membership_proof(idx);
                let csmt_proof = csmt.generate_membership_proof(idx);
                for level in 0..3 {
                    assert_eq!(smt_proof.path[level], csmt_proof.path[level]);
                    assert_eq!(smt_proof.direction_bits[level], csmt_proof.direction_bits[level]);
                }
                assert!(csmt_proof.check_membership(&csmt.root(), &empty_val, &poseidon).unwrap());
            }
        }

        // Height 10, 20 random leaves, 10 non-existent probes
        {
            let leaves: BTreeMap<u32, Fp> =
                (0..20).map(|i| (i * 50, Fp::random(rng))).collect();
            let (smt, csmt) = create_both::<10>(&leaves);
            let non_existent = [1u64, 2, 49, 51, 99, 101, 200, 500, 800, 1023];
            for &idx in &non_existent {
                let smt_proof = smt.generate_membership_proof(idx);
                let csmt_proof = csmt.generate_membership_proof(idx);
                for level in 0..10 {
                    assert_eq!(
                        smt_proof.path[level], csmt_proof.path[level],
                        "H10 path mismatch at level {} for index {}",
                        level, idx
                    );
                    assert_eq!(
                        smt_proof.direction_bits[level], csmt_proof.direction_bits[level],
                        "H10 dir mismatch at level {} for index {}",
                        level, idx
                    );
                }
                assert!(csmt_proof.check_membership(&csmt.root(), &empty_val, &poseidon).unwrap());
            }
        }

        // Height 20, sparse leaves
        {
            let indices = [0u32, 100, 500, 10_000, 100_000, 500_000];
            let leaves: BTreeMap<u32, Fp> =
                indices.iter().map(|&i| (i, Fp::random(rng))).collect();
            let (smt, csmt) = create_both::<20>(&leaves);
            let non_existent = [1u64, 50, 99, 101, 250, 501, 9999, 10001, 99999, 999_999];
            for &idx in &non_existent {
                let smt_proof = smt.generate_membership_proof(idx);
                let csmt_proof = csmt.generate_membership_proof(idx);
                for level in 0..20 {
                    assert_eq!(
                        smt_proof.path[level], csmt_proof.path[level],
                        "H20 path mismatch at level {} for index {}",
                        level, idx
                    );
                    assert_eq!(
                        smt_proof.direction_bits[level], csmt_proof.direction_bits[level],
                        "H20 dir mismatch at level {} for index {}",
                        level, idx
                    );
                }
                assert!(csmt_proof.check_membership(&csmt.root(), &empty_val, &poseidon).unwrap());
            }
        }
    }

    #[test]
    fn test_nonexistent_sparse_proof() {
        // Verify sparse proofs for non-existent indices match the
        // original SMT and produce the correct full root.
        let rng = OsRng;
        let empty_leaf = [0u8; 64];
        let poseidon = Poseidon2::<Fp, 2>::new();
        let empty_val = Fp::from_uniform_bytes(&empty_leaf);

        let indices = [0u32, 5, 13, 100, 500, 1000];
        let leaves: BTreeMap<u32, Fp> =
            indices.iter().map(|&i| (i, Fp::random(rng))).collect();
        let (smt, csmt) = create_both::<20>(&leaves);

        let non_existent = [1u64, 4, 6, 12, 14, 50, 101, 501, 999, 1001];
        for &idx in &non_existent {
            let smt_sparse = smt.generate_sparse_membership_proof(idx);
            let csmt_sparse = csmt.generate_sparse_membership_proof(idx);

            assert_eq!(
                smt_sparse.entries.len(),
                csmt_sparse.entries.len(),
                "sparse entry count mismatch for non-existent index {}",
                idx
            );

            for (i, (s, c)) in smt_sparse
                .entries
                .iter()
                .zip(csmt_sparse.entries.iter())
                .enumerate()
            {
                assert_eq!(
                    s.sibling, c.sibling,
                    "sparse sibling mismatch at entry {} for index {}",
                    i, idx
                );
                assert_eq!(
                    s.direction_bit, c.direction_bit,
                    "sparse direction_bit mismatch at entry {} for index {}",
                    i, idx
                );
                assert_eq!(
                    s.level, c.level,
                    "sparse level mismatch at entry {} for index {}",
                    i, idx
                );
            }

            // Full root should match
            let full_root = csmt_sparse
                .calculate_full_root(&empty_val, &poseidon, idx)
                .unwrap();
            assert_eq!(
                full_root,
                csmt.root(),
                "sparse full root mismatch for non-existent index {}",
                idx
            );
        }
    }

    // ---- Benchmarks ----

    #[test]
    fn test_bench_existing_vs_nonexistent() {
        use std::time::Instant;

        let rng = OsRng;
        let hasher = Poseidon2::<Fp, 2>::new();
        let empty_leaf = [0u8; 64];

        // Height 10, 50 leaves
        {
            let leaves: BTreeMap<u32, Fp> =
                (0..50).map(|i| (i, Fp::random(rng))).collect();
            let csmt = CompressedSMT::<Fp, TestHasher, 10>::new_from_u32(
                &leaves, &hasher, &empty_leaf,
            )
            .unwrap();

            let existing_indices: Vec<u64> = leaves.keys().map(|&k| k as u64).collect();
            let non_existent: Vec<u64> = (50..150).map(|i| i as u64).collect();

            let iters = 10;

            let start = Instant::now();
            for _ in 0..iters {
                for &idx in &existing_indices[..10] {
                    let _ = csmt.generate_membership_proof(idx);
                }
            }
            let existing_dense = start.elapsed();

            let start = Instant::now();
            for _ in 0..iters {
                for &idx in &non_existent[..10] {
                    let _ = csmt.generate_membership_proof(idx);
                }
            }
            let nonexist_dense = start.elapsed();

            let start = Instant::now();
            for _ in 0..iters {
                for &idx in &existing_indices[..10] {
                    let _ = csmt.generate_sparse_membership_proof(idx);
                }
            }
            let existing_sparse = start.elapsed();

            let start = Instant::now();
            for _ in 0..iters {
                for &idx in &non_existent[..10] {
                    let _ = csmt.generate_sparse_membership_proof(idx);
                }
            }
            let nonexist_sparse = start.elapsed();

            eprintln!("\n--- Height 10 benchmark ({} iters x 10 proofs) ---", iters);
            eprintln!(
                "  Dense:  existing {:?}  |  non-existent {:?}",
                existing_dense, nonexist_dense
            );
            eprintln!(
                "  Sparse: existing {:?}  |  non-existent {:?}",
                existing_sparse, nonexist_sparse
            );
        }

        // Height 20, 100 sparse leaves (only CompressedSMT, skips slow old SMT build)
        {
            let leaves: BTreeMap<u32, Fp> =
                (0..100).map(|i| (i * 100, Fp::random(rng))).collect();
            let csmt = CompressedSMT::<Fp, TestHasher, 20>::new_from_u32(
                &leaves, &hasher, &empty_leaf,
            )
            .unwrap();

            let existing_indices: Vec<u64> = leaves.keys().take(10).map(|&k| k as u64).collect();
            let non_existent: Vec<u64> = (1..11).map(|i| i as u64).collect();

            let iters = 5;

            let start = Instant::now();
            for _ in 0..iters {
                for &idx in &existing_indices {
                    let _ = csmt.generate_membership_proof(idx);
                }
            }
            let existing_dense = start.elapsed();

            let start = Instant::now();
            for _ in 0..iters {
                for &idx in &non_existent {
                    let _ = csmt.generate_membership_proof(idx);
                }
            }
            let nonexist_dense = start.elapsed();

            let start = Instant::now();
            for _ in 0..iters {
                for &idx in &existing_indices {
                    let _ = csmt.generate_sparse_membership_proof(idx);
                }
            }
            let existing_sparse = start.elapsed();

            let start = Instant::now();
            for _ in 0..iters {
                for &idx in &non_existent {
                    let _ = csmt.generate_sparse_membership_proof(idx);
                }
            }
            let nonexist_sparse = start.elapsed();

            eprintln!("\n--- Height 20 benchmark ({} iters x 10 proofs) ---", iters);
            eprintln!(
                "  Dense:  existing {:?}  |  non-existent {:?}",
                existing_dense, nonexist_dense
            );
            eprintln!(
                "  Sparse: existing {:?}  |  non-existent {:?}",
                existing_sparse, nonexist_sparse
            );
        }
    }
}
