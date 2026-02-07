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

/// Minimum number of leaves in a subtree to trigger parallel recursion via rayon.
const PARALLEL_THRESHOLD: usize = 1024;

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
) -> CompressedNode<F> {
    match leaves.len() {
        0 => CompressedNode::Zero,
        1 => {
            let hash = compute_single_hash(leaves[0], level, hasher, empty_hashes);
            CompressedNode::Single {
                leaf_index: leaves[0].0,
                leaf_value: leaves[0].1,
                hash,
            }
        }
        2 => {
            let hash =
                compute_double_hash(leaves[0], leaves[1], level, hasher, empty_hashes);
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
                    || build(left_leaves, level - 1, hasher, empty_hashes),
                    || build(right_leaves, level - 1, hasher, empty_hashes),
                )
            } else {
                (
                    build(left_leaves, level - 1, hasher, empty_hashes),
                    build(right_leaves, level - 1, hasher, empty_hashes),
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
/// of the N levels. Panics if the target leaf is not in the tree.
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
            panic!(
                "target leaf index {} not found in tree (hit Zero node)",
                target_index
            );
        }
        CompressedNode::Single {
            leaf_index,
            leaf_value,
            ..
        } => {
            assert_eq!(
                *leaf_index, target_index,
                "Single node leaf index {} doesn't match target {}",
                leaf_index, target_index
            );
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
        }
        CompressedNode::Double { leaf_a, leaf_b, .. } => {
            let (target_leaf, other_leaf) = if leaf_a.0 == target_index {
                (leaf_a, leaf_b)
            } else {
                assert_eq!(
                    leaf_b.0, target_index,
                    "Double node doesn't contain target leaf {}",
                    target_index
                );
                (leaf_b, leaf_a)
            };

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
/// Entries are naturally produced in ascending level order (leaf → root).
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
            panic!(
                "target leaf index {} not found in tree (hit Zero node)",
                target_index
            );
        }
        CompressedNode::Single { leaf_index, .. } => {
            assert_eq!(
                *leaf_index, target_index,
                "Single node leaf index {} doesn't match target {}",
                leaf_index, target_index
            );
            // All siblings within a Single node are empty hashes — no entries to add.
        }
        CompressedNode::Double { leaf_a, leaf_b, .. } => {
            let other = if leaf_a.0 == target_index {
                leaf_b
            } else {
                assert_eq!(
                    leaf_b.0, target_index,
                    "Double node doesn't contain target leaf {}",
                    target_index
                );
                leaf_a
            };

            let xor = target_index ^ other.0;
            let div_level = (64 - xor.leading_zeros()) as usize;

            // The other leaf appears as a non-empty sibling at the divergence level.
            let sibling_hash =
                compute_single_hash(*other, div_level - 1, hasher, empty_hashes);
            let bit = (target_index >> (div_level - 1)) & 1;

            entries.push(SparsePathEntry {
                sibling: sibling_hash,
                direction_bit: bit == 1,
                level: div_level - 1,
            });
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
        let empty_hashes = gen_empty_hashes::<F, H, N>(hasher, empty_leaf)?;
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

        let root = build(&sorted_leaves, N, hasher, &empty_hashes);

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
    /// # Panics
    ///
    /// Panics if `index` is not a leaf in this tree.
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
    /// # Panics
    ///
    /// Panics if `index` is not a leaf in this tree.
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
    use crate::poseidon::Poseidon;
    use crate::smt::SparseMerkleTree;
    use ff::Field;
    use pasta_curves::Fp;
    use rand::rngs::OsRng;
    use rand::Rng;
    use std::collections::BTreeMap;

    type TestHasher = Poseidon<Fp, 2>;

    /// Helper: create both the original SMT and compressed SMT from the same leaves.
    fn create_both<const N: usize>(
        leaves: &BTreeMap<u32, Fp>,
    ) -> (
        SparseMerkleTree<Fp, TestHasher, N>,
        CompressedSMT<Fp, TestHasher, N>,
    ) {
        let hasher = Poseidon::<Fp, 2>::new();
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
        let hasher = Poseidon::<Fp, 2>::new();
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
        let poseidon = Poseidon::<Fp, 2>::new();
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
        let poseidon = Poseidon::<Fp, 2>::new();

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
        let poseidon = Poseidon::<Fp, 2>::new();
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
        let poseidon = Poseidon::<Fp, 2>::new();
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
        let poseidon = Poseidon::<Fp, 2>::new();
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
        let poseidon = Poseidon::<Fp, 2>::new();
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
}
