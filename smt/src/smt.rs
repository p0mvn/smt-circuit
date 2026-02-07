// This file is adapted from Webb and Arkworks:
// https://github.com/webb-tools/arkworks-gadgets

// Copyright (C) 2021 Webb Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

// Copyright (c) zkMove Authors
// SPDX-License-Identifier: Apache-2.0

//! This file provides a native implementation of the Sparse Merkle tree data
//! structure.
//!
//! A Sparse Merkle tree is a type of Merkle tree, but it is much easier to
//! prove non-membership in a sparse Merkle tree than in an arbitrary Merkle
//! tree. For an explanation of sparse Merkle trees, see:
//! `<https://medium.com/@kelvinfichter/whats-a-sparse-merkle-tree-acda70aeb837>`
//!
//! In this file we define the `Path` and `SparseMerkleTree` structs.
//! These depend on your choice of a prime field F, a field hasher over F
//! (any hash function that maps F^2 to F will do, e.g. the poseidon hash
//! function of width 3 where an input of zero is used for padding), and the
//! height N of the sparse Merkle tree.
//!
//! The path corresponding to a given leaf node is stored as an N-tuple of pairs
//! of field elements. Each pair consists of a node lying on the path from the
//! leaf node to the root, and that node's sibling.  For example, suppose
//! ```text
//!           a
//!         /   \
//!        b     c
//!       / \   / \
//!      d   e f   g
//! ```
//! is our Sparse Merkle tree, and `a` through `g` are field elements stored at
//! the nodes. Then the merkle proof path `e-b-a` from leaf `e` to root `a` is
//! stored as `[(d,e), (b,c)]`

use crate::poseidon::FieldHasher;
use anyhow::{Error, Result};
use ff::{FromUniformBytes, PrimeField};
use std::{
    borrow::ToOwned,
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
};

/// Error enum for Sparse Merkle Tree.
#[derive(Debug)]
pub enum MerkleError {
    /// Thrown when the given leaf is not in the tree or the path.
    InvalidLeaf,
    /// Thrown when the merkle path is invalid.
    InvalidPathNodes,
}

impl core::fmt::Display for MerkleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            MerkleError::InvalidLeaf => "Invalid leaf".to_owned(),
            MerkleError::InvalidPathNodes => "Path nodes are not consistent".to_owned(),
        };
        write!(f, "{}", msg)
    }
}

impl std::error::Error for MerkleError {}

/// The Path struct.
///
/// The path contains a sequence of sibling nodes that make up a merkle proof.
/// Each pair is used to identify whether an incremental merkle root
/// construction is valid at each intermediate step.
#[derive(Clone)]
pub struct Path<F: PrimeField, H: FieldHasher<F, 2>, const N: usize> {
    /// The path represented as a sequence of sibling pairs.
    pub path: [(F, F); N],
    /// Direction bits indicating which side of the tree the path descends on each level.
    /// `false` (0) means the node is a left child, `true` (1) means it is a right child.
    pub direction_bits: [bool; N],
    /// The phantom hasher type used to reconstruct the merkle root.
    pub marker: PhantomData<H>,
}

impl<F: PrimeField, H: FieldHasher<F, 2>, const N: usize> Path<F, H, N> {
    /// Takes in an expected `root_hash` and leaf-level data (i.e. hashes of
    /// secrets) for a leaf and checks that the leaf belongs to a tree having
    /// the expected hash.
    pub fn check_membership(&self, root_hash: &F, leaf: &F, hasher: &H) -> Result<bool, Error> {
        let root = self.calculate_root(leaf, hasher)?;
        Ok(root == *root_hash)
    }

    /// Assumes leaf contains leaf-level data, i.e. hashes of secrets
    /// stored on leaf-level.
    pub fn calculate_root(&self, leaf: &F, hasher: &H) -> Result<F, Error> {
        if *leaf != self.path[0].0 && *leaf != self.path[0].1 {
            return Err(MerkleError::InvalidLeaf.into());
        }

        let mut prev = *leaf;
        // Check levels between leaf level and root
        for &(ref left_hash, ref right_hash) in &self.path {
            if &prev != left_hash && &prev != right_hash {
                return Err(MerkleError::InvalidPathNodes.into());
            }
            prev = hasher.hash([*left_hash, *right_hash])?;
        }

        Ok(prev)
    }

    /// Given leaf data determine what the index of this leaf must be
    /// in the Merkle tree it belongs to.  Before doing so check that the leaf
    /// does indeed belong to a tree with the given `root_hash`
    pub fn get_index(&self, root_hash: &F, leaf: &F, hasher: &H) -> Result<F, Error> {
        if !self.check_membership(root_hash, leaf, hasher)? {
            return Err(MerkleError::InvalidLeaf.into());
        }

        let mut prev = *leaf;
        let mut index = F::ZERO;
        let mut twopower = F::ONE;
        // Check levels between leaf level and root
        for &(ref left_hash, ref right_hash) in &self.path {
            // Check if the previous hash is for a left node or right node
            if &prev != left_hash {
                index += twopower;
            }
            twopower = twopower + twopower;
            prev = hasher.hash([*left_hash, *right_hash])?;
        }

        Ok(index)
    }
}

/// A single non-zero sibling entry in a sparse path.
#[derive(Clone, Debug)]
pub struct SparsePathEntry<F: PrimeField> {
    /// The sibling hash at this level.
    pub sibling: F,
    /// Direction bit: false = left child, true = right child.
    pub direction_bit: bool,
    /// The level in the tree (0 = leaf level).
    pub level: usize,
}

/// Variable-length Merkle path containing only non-zero siblings.
///
/// Instead of storing all N levels (most of which have empty-hash siblings
/// in a sparse tree), this stores only the K levels where the sibling differs
/// from the precomputed empty hash. This enables a much smaller circuit.
pub struct SparsePath<F: PrimeField, H: FieldHasher<F, 2>> {
    /// The non-zero sibling entries, ordered from leaf to root.
    pub entries: Vec<SparsePathEntry<F>>,
    /// The height of the tree (number of levels).
    pub tree_height: usize,
    /// Precomputed empty hashes for each level.
    pub empty_hashes: Vec<F>,
    /// Phantom data for the hasher type.
    pub marker: PhantomData<H>,
}

// Manual Clone impl to avoid requiring H: Clone (H is only in PhantomData).
impl<F: PrimeField, H: FieldHasher<F, 2>> Clone for SparsePath<F, H> {
    fn clone(&self) -> Self {
        SparsePath {
            entries: self.entries.clone(),
            tree_height: self.tree_height,
            empty_hashes: self.empty_hashes.clone(),
            marker: PhantomData,
        }
    }
}

impl<F: PrimeField, H: FieldHasher<F, 2>> SparsePath<F, H> {
    /// Hash through only the K non-zero entries to produce the compact root.
    /// This matches exactly what the circuit computes.
    pub fn calculate_compact_root(&self, leaf: &F, hasher: &H) -> Result<F, Error> {
        let mut prev = *leaf;
        for entry in &self.entries {
            let (left, right) = if entry.direction_bit {
                (entry.sibling, prev) // we're right child, sibling is left
            } else {
                (prev, entry.sibling) // we're left child, sibling is right
            };
            prev = hasher.hash([left, right])?;
        }
        Ok(prev)
    }

    /// Hash through all tree_height levels, using empty_hashes for gap levels.
    /// Returns the standard SMT root.
    pub fn calculate_full_root(
        &self,
        leaf: &F,
        hasher: &H,
        leaf_index: u64,
    ) -> Result<F, Error> {
        let mut prev = *leaf;
        let mut entry_idx = 0;
        for level in 0..self.tree_height {
            let direction_bit = (leaf_index >> level) & 1 == 1;
            let sibling = if entry_idx < self.entries.len()
                && self.entries[entry_idx].level == level
            {
                let s = self.entries[entry_idx].sibling;
                entry_idx += 1;
                s
            } else {
                self.empty_hashes[level]
            };
            let (left, right) = if direction_bit {
                (sibling, prev)
            } else {
                (prev, sibling)
            };
            prev = hasher.hash([left, right])?;
        }
        Ok(prev)
    }

    /// Verify that the compact root chains through gap levels to produce
    /// the expected standard root.
    pub fn verify_against_root(
        &self,
        leaf: &F,
        hasher: &H,
        leaf_index: u64,
        expected_root: &F,
    ) -> Result<bool, Error> {
        let full_root = self.calculate_full_root(leaf, hasher, leaf_index)?;
        Ok(full_root == *expected_root)
    }

    /// Convert to fixed-size arrays for the circuit.
    /// Returns (siblings, direction_bits, is_active) each of length MAX_K,
    /// padded with zeros for inactive slots.
    pub fn to_padded_arrays<const MAX_K: usize>(
        &self,
    ) -> ([F; MAX_K], [bool; MAX_K], [bool; MAX_K]) {
        assert!(
            self.entries.len() <= MAX_K,
            "SparsePath has {} entries but MAX_K is {}",
            self.entries.len(),
            MAX_K,
        );
        let mut siblings = [F::ZERO; MAX_K];
        let mut direction_bits = [false; MAX_K];
        let mut is_active = [false; MAX_K];
        for (i, entry) in self.entries.iter().enumerate() {
            siblings[i] = entry.sibling;
            direction_bits[i] = entry.direction_bit;
            is_active[i] = true;
        }
        (siblings, direction_bits, is_active)
    }
}

/// The Sparse Merkle Tree struct.
///
/// The Sparse Merkle Tree stores a set of leaves represented in a map and
/// a set of empty hashes that it uses to represent the sparse areas of the
/// tree.
pub struct SparseMerkleTree<F: PrimeField + FromUniformBytes<64>, H: FieldHasher<F, 2>, const N: usize> {
    /// A map from leaf indices to leaf data stored as field elements.
    pub tree: BTreeMap<u64, F>,
    /// An array of default hashes hashed with themselves `N` times.
    empty_hashes: [F; N],
    /// The phantom hasher type used to build the merkle tree.
    marker: PhantomData<H>,
}

impl<F: PrimeField + FromUniformBytes<64>, H: FieldHasher<F, 2>, const N: usize> SparseMerkleTree<F, H, N> {
    /// Takes a batch of field elements, inserts
    /// these hashes into the tree, and updates the merkle root.
    pub fn insert_batch(&mut self, leaves: &BTreeMap<u32, F>, hasher: &H) -> Result<(), Error> {
        let last_level_index: u64 = (1u64 << N) - 1;

        let mut level_idxs: BTreeSet<u64> = BTreeSet::new();
        for (i, leaf) in leaves {
            let true_index = last_level_index + (*i as u64);
            self.tree.insert(true_index, *leaf);
            level_idxs.insert(parent(true_index).unwrap());
        }

        for level in 0..N {
            log::debug!(
                "[SMT] Processing level {}/{} ({} nodes to hash)",
                level + 1,
                N,
                level_idxs.len()
            );
            let level_start = Instant::now();
            let mut new_idxs: BTreeSet<u64> = BTreeSet::new();
            let empty_hash = self.empty_hashes[level];
            for i in level_idxs {
                let left_index = left_child(i);
                let right_index = right_child(i);
                let left = self.tree.get(&left_index).unwrap_or(&empty_hash);
                let right = self.tree.get(&right_index).unwrap_or(&empty_hash);
                self.tree.insert(i, hasher.hash([*left, *right])?);

                let parent = match parent(i) {
                    Some(i) => i,
                    None => break,
                };
                new_idxs.insert(parent);
            }
            log::debug!(
                "[SMT] Level {}/{} completed in {:?}",
                level + 1,
                N,
                level_start.elapsed()
            );
            level_idxs = new_idxs;
        }

        Ok(())
    }

    /// Creates a new Sparse Merkle Tree from a map of indices to field
    /// elements.
    pub fn new(
        leaves: &BTreeMap<u32, F>,
        hasher: &H,
        empty_leaf: &[u8; 64],
    ) -> Result<Self, Error> {
        log::info!(
            "[SMT] Building tree: height={}, leaves={}",
            N,
            leaves.len()
        );
        let start = Instant::now();

        // Ensure the tree can hold this many leaves
        let last_level_size = leaves.len().next_power_of_two();
        let tree_size = 2 * last_level_size - 1;
        let tree_height = tree_height(tree_size as u64);
        assert!(tree_height <= N as u32);

        // Initialize the merkle tree
        let tree: BTreeMap<u64, F> = BTreeMap::new();
        let empty_hashes = gen_empty_hashes(hasher, empty_leaf)?;

        let mut smt = SparseMerkleTree::<F, H, N> {
            tree,
            empty_hashes,
            marker: PhantomData,
        };
        smt.insert_batch(leaves, hasher)?;

        log::info!(
            "[SMT] Tree built in {:?} (height={}, leaves={}, tree_nodes={})",
            start.elapsed(),
            N,
            leaves.len(),
            smt.tree.len()
        );

        Ok(smt)
    }

    /// Creates a new Sparse Merkle Tree from an array of field elements.
    pub fn new_sequential(leaves: &[F], hasher: &H, empty_leaf: &[u8; 64]) -> Result<Self, Error> {
        log::info!(
            "[SMT] Building sequential tree: height={}, leaves={}",
            N,
            leaves.len()
        );
        let pairs: BTreeMap<u32, F> = leaves
            .iter()
            .enumerate()
            .map(|(i, l)| (i as u32, *l))
            .collect();
        let smt = Self::new(&pairs, hasher, empty_leaf)?;

        Ok(smt)
    }

    /// Returns the Merkle tree root.
    pub fn root(&self) -> F {
        self.tree
            .get(&0)
            .cloned()
            .unwrap_or(*self.empty_hashes.last().unwrap())
    }

    /// Give the path leading from the leaf at `index` up to the root.  This is
    /// a "proof" in the sense of "valid path in a Merkle tree", not a ZK
    /// argument.
    pub fn generate_membership_proof(&self, index: u64) -> Path<F, H, N> {
        let mut path = [(F::ZERO, F::ZERO); N];
        let mut direction_bits = [false; N];

        let tree_index = convert_index_to_last_level(index, N);

        // Iterate from the leaf up to the root, storing all intermediate hash values.
        let mut current_node = tree_index;
        let mut level = 0;
        while !is_root(current_node) {
            let sibling_node = sibling(current_node).unwrap();

            let empty_hash = &self.empty_hashes[level];

            let current = self.tree.get(&current_node).cloned().unwrap_or(*empty_hash);
            let sibling = self.tree.get(&sibling_node).cloned().unwrap_or(*empty_hash);

            // direction_bit = true means the node is a right child
            direction_bits[level] = !is_left_child(current_node);

            if is_left_child(current_node) {
                path[level] = (current, sibling);
            } else {
                path[level] = (sibling, current);
            }
            current_node = parent(current_node).unwrap();
            level += 1;
        }

        Path {
            path,
            direction_bits,
            marker: PhantomData,
        }
    }

    /// Generate a sparse membership proof containing only non-zero sibling levels.
    ///
    /// This walks from the leaf to the root and only includes levels where
    /// the sibling hash differs from the precomputed empty hash at that level.
    /// The resulting `SparsePath` has K entries where K << N for sparse trees.
    pub fn generate_sparse_membership_proof(&self, index: u64) -> SparsePath<F, H> {
        let tree_index = convert_index_to_last_level(index, N);
        let mut entries = Vec::new();

        let mut current_node = tree_index;
        let mut level = 0;
        while !is_root(current_node) {
            let sibling_node = sibling(current_node).unwrap();
            let empty_hash = &self.empty_hashes[level];
            let sibling_val = self
                .tree
                .get(&sibling_node)
                .cloned()
                .unwrap_or(*empty_hash);

            // Only include levels where sibling differs from empty hash
            if sibling_val != *empty_hash {
                entries.push(SparsePathEntry {
                    sibling: sibling_val,
                    direction_bit: !is_left_child(current_node),
                    level,
                });
            }

            current_node = parent(current_node).unwrap();
            level += 1;
        }

        SparsePath {
            entries,
            tree_height: N,
            empty_hashes: self.empty_hashes.to_vec(),
            marker: PhantomData,
        }
    }
}

/// A function to generate empty hashes with a given `default_leaf`.
///
/// Given a `FieldHasher`, generate a list of `N` hashes consisting
/// of the `default_leaf` hashed with itself and repeated `N` times
/// with the intermediate results. These are used to initialize the
/// sparse portion of the Sparse Merkle Tree.
pub fn gen_empty_hashes<F: PrimeField + FromUniformBytes<64>, H: FieldHasher<F, 2>, const N: usize>(
    hasher: &H,
    default_leaf: &[u8; 64],
) -> Result<[F; N], Error> {
    let mut empty_hashes = [F::ZERO; N];

    // Convert default_leaf bytes to field element using from_uniform_bytes
    let mut empty_hash = F::from_uniform_bytes(default_leaf);
    for item in empty_hashes.iter_mut().take(N) {
        *item = empty_hash;
        empty_hash = hasher.hash([empty_hash, empty_hash])?;
    }

    Ok(empty_hashes)
}

fn convert_index_to_last_level(index: u64, height: usize) -> u64 {
    index + (1u64 << height) - 1
}

/// Returns the log2 value of the given number.
#[inline]
fn log2(number: u64) -> u32 {
    ark_std::log2(number as usize)
}

/// Returns the height of the tree, given the size of the tree.
#[inline]
fn tree_height(tree_size: u64) -> u32 {
    log2(tree_size)
}

/// Returns true iff the index represents the root.
#[inline]
fn is_root(index: u64) -> bool {
    index == 0
}

/// Returns the index of the left child, given an index.
#[inline]
fn left_child(index: u64) -> u64 {
    2 * index + 1
}

/// Returns the index of the right child, given an index.
#[inline]
fn right_child(index: u64) -> u64 {
    2 * index + 2
}

/// Returns the index of the sibling, given an index.
#[inline]
fn sibling(index: u64) -> Option<u64> {
    if index == 0 {
        None
    } else if is_left_child(index) {
        Some(index + 1)
    } else {
        Some(index - 1)
    }
}

/// Returns true iff the given index represents a left child.
#[inline]
fn is_left_child(index: u64) -> bool {
    index % 2 == 1
}

/// Returns the index of the parent, given an index.
#[inline]
fn parent(index: u64) -> Option<u64> {
    if index > 0 {
        Some((index - 1) >> 1)
    } else {
        None
    }
}

#[cfg(test)]
mod test {
    use super::{gen_empty_hashes, SparseMerkleTree};
    use crate::poseidon::{FieldHasher, Poseidon};
    use ff::{Field, FromUniformBytes, PrimeField};
    use pasta_curves::Fp;
    use rand::rngs::OsRng;
    use std::collections::BTreeMap;

    //helper to change leaves array to BTreeMap and then create SMT
    fn create_merkle_tree<F: PrimeField + FromUniformBytes<64> + Ord, H: FieldHasher<F, 2>, const N: usize>(
        hasher: H,
        leaves: &[F],
        default_leaf: &[u8; 64],
    ) -> SparseMerkleTree<F, H, N> {
        let pairs: BTreeMap<u32, F> = leaves
            .iter()
            .enumerate()
            .map(|(i, l)| (i as u32, *l))
            .collect();

        SparseMerkleTree::<F, H, N>::new(&pairs, &hasher, default_leaf).unwrap()
    }

    #[test]
    fn should_create_tree_poseidon() {
        let poseidon = Poseidon::<Fp, 2>::new();
        let default_leaf = [0u8; 64];
        let rng = OsRng;
        let leaves = [Fp::random(rng), Fp::random(rng), Fp::random(rng)];
        const HEIGHT: usize = 3;
        let smt = create_merkle_tree::<Fp, Poseidon<Fp, 2>, HEIGHT>(
            poseidon.clone(),
            &leaves,
            &default_leaf,
        );

        let root = smt.root();

        let empty_hashes =
            gen_empty_hashes::<Fp, Poseidon<Fp, 2>, HEIGHT>(&poseidon, &default_leaf).unwrap();
        let hash1 = leaves[0];
        let hash2 = leaves[1];
        let hash3 = leaves[2];

        let hash12 = poseidon.hash([hash1, hash2]).unwrap();
        let hash34 = poseidon.hash([hash3, empty_hashes[0]]).unwrap();

        let hash1234 = poseidon.hash([hash12, hash34]).unwrap();
        let calc_root = poseidon.hash([hash1234, empty_hashes[2]]).unwrap();

        assert_eq!(root, calc_root);
    }

    #[test]
    fn should_generate_and_validate_proof_poseidon() {
        let poseidon = Poseidon::<Fp, 2>::new();
        let default_leaf = [0u8; 64];
        let rng = OsRng;
        let leaves = [Fp::random(rng), Fp::random(rng), Fp::random(rng)];
        const HEIGHT: usize = 3;
        let smt = create_merkle_tree::<Fp, Poseidon<Fp, 2>, HEIGHT>(
            poseidon.clone(),
            &leaves,
            &default_leaf,
        );

        let proof = smt.generate_membership_proof(0);

        let res = proof
            .check_membership(&smt.root(), &leaves[0], &poseidon)
            .unwrap();
        assert!(res);
    }

    #[test]
    fn should_find_the_index_poseidon() {
        let poseidon = Poseidon::<Fp, 2>::new();
        let default_leaf = [0u8; 64];
        let rng = OsRng;
        let leaves = [Fp::random(rng), Fp::random(rng), Fp::random(rng)];
        const HEIGHT: usize = 3;
        let smt = create_merkle_tree::<Fp, Poseidon<Fp, 2>, HEIGHT>(
            poseidon.clone(),
            &leaves,
            &default_leaf,
        );

        let index = 2;

        let proof = smt.generate_membership_proof(index);

        let res = proof
            .get_index(&smt.root(), &leaves[index as usize], &poseidon)
            .unwrap();
        let desired_res = Fp::from(index);

        assert_eq!(res, desired_res);
    }

    // ========== Sparse Path Tests ==========

    #[test]
    fn test_sparse_proof_generation() {
        let poseidon = Poseidon::<Fp, 2>::new();
        let default_leaf = [0u8; 64];
        let rng = OsRng;
        let leaves = [Fp::random(rng), Fp::random(rng), Fp::random(rng)];
        const HEIGHT: usize = 20;
        let smt = create_merkle_tree::<Fp, Poseidon<Fp, 2>, HEIGHT>(
            poseidon.clone(),
            &leaves,
            &default_leaf,
        );

        let sparse_proof = smt.generate_sparse_membership_proof(0);

        // K should be much less than N for a sparse tree
        println!(
            "Sparse proof entries: {} out of {} levels",
            sparse_proof.entries.len(),
            HEIGHT
        );
        assert!(sparse_proof.entries.len() < HEIGHT);
        assert!(!sparse_proof.entries.is_empty());

        // Entries should be in ascending level order
        for i in 1..sparse_proof.entries.len() {
            assert!(sparse_proof.entries[i].level > sparse_proof.entries[i - 1].level);
        }
    }

    #[test]
    fn test_compact_root_matches() {
        let poseidon = Poseidon::<Fp, 2>::new();
        let default_leaf = [0u8; 64];
        let rng = OsRng;
        let leaves = [Fp::random(rng), Fp::random(rng), Fp::random(rng)];
        const HEIGHT: usize = 20;
        let smt = create_merkle_tree::<Fp, Poseidon<Fp, 2>, HEIGHT>(
            poseidon.clone(),
            &leaves,
            &default_leaf,
        );

        let sparse_proof = smt.generate_sparse_membership_proof(0);
        let compact_root = sparse_proof
            .calculate_compact_root(&leaves[0], &poseidon)
            .unwrap();

        // Compute the same thing manually
        let mut prev = leaves[0];
        for entry in &sparse_proof.entries {
            let (left, right) = if entry.direction_bit {
                (entry.sibling, prev)
            } else {
                (prev, entry.sibling)
            };
            prev = poseidon.hash([left, right]).unwrap();
        }
        assert_eq!(compact_root, prev);
    }

    #[test]
    fn test_full_root_matches_standard() {
        let poseidon = Poseidon::<Fp, 2>::new();
        let default_leaf = [0u8; 64];
        let rng = OsRng;
        let leaves = [Fp::random(rng), Fp::random(rng), Fp::random(rng)];
        const HEIGHT: usize = 20;
        let smt = create_merkle_tree::<Fp, Poseidon<Fp, 2>, HEIGHT>(
            poseidon.clone(),
            &leaves,
            &default_leaf,
        );

        for index in 0..3u64 {
            let sparse_proof = smt.generate_sparse_membership_proof(index);
            let full_root = sparse_proof
                .calculate_full_root(&leaves[index as usize], &poseidon, index)
                .unwrap();
            assert_eq!(
                full_root,
                smt.root(),
                "Full root mismatch for leaf index {}",
                index
            );
        }
    }

    #[test]
    fn test_sparse_roundtrip() {
        let poseidon = Poseidon::<Fp, 2>::new();
        let default_leaf = [0u8; 64];
        let rng = OsRng;
        let leaves = [Fp::random(rng), Fp::random(rng), Fp::random(rng)];
        const HEIGHT: usize = 20;
        let smt = create_merkle_tree::<Fp, Poseidon<Fp, 2>, HEIGHT>(
            poseidon.clone(),
            &leaves,
            &default_leaf,
        );

        for index in 0..3u64 {
            let sparse_proof = smt.generate_sparse_membership_proof(index);

            // verify_against_root should pass
            let result = sparse_proof
                .verify_against_root(&leaves[index as usize], &poseidon, index, &smt.root())
                .unwrap();
            assert!(result, "Sparse roundtrip failed for leaf index {}", index);

            // Verify that to_padded_arrays works
            let (siblings, dir_bits, is_active) = sparse_proof.to_padded_arrays::<32>();
            let k = sparse_proof.entries.len();
            for i in 0..k {
                assert!(is_active[i]);
                assert_eq!(siblings[i], sparse_proof.entries[i].sibling);
                assert_eq!(dir_bits[i], sparse_proof.entries[i].direction_bit);
            }
            for i in k..32 {
                assert!(!is_active[i]);
                assert_eq!(siblings[i], Fp::ZERO);
            }
        }
    }
}
