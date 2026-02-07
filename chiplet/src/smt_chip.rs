// This file is adapted from Webb and Arkworks:
// https://github.com/webb-tools/arkworks-gadgets

// Copyright (C) 2021 Webb Technologies Inc.
// SPDX-License-Identifier: Apache-2.0

// Copyright (c) zkMove Authors
// SPDX-License-Identifier: Apache-2.0

//! A Plonk gadget implementation of the Sparse Merkle Tree data structure.
//! For more info on the Sparse Merkle Tree data structure, see the
//! documentation for the native implementation.

use crate::poseidon2_chip::{Poseidon2Chip, Poseidon2Config};
use crate::utilities::{
    ConditionalSelectChip, ConditionalSelectConfig, ConditionalSwapChip, ConditionalSwapConfig,
    IsEqualChip, IsEqualConfig, NUM_OF_SWAP_ADVICE_COLUMNS, NUM_OF_UTILITY_ADVICE_COLUMNS,
};
use ff::PrimeField;
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error, Selector},
};
use smt::poseidon::FieldHasher;
use smt::smt::{Path, SparsePath};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct PathConfig<F: PrimeField, const N: usize> {
    s_path: Selector,
    advices: [Column<Advice>; N],
    poseidon_config: Poseidon2Config<F>,
    is_eq_config: IsEqualConfig<F>,
    swap_config: ConditionalSwapConfig<F>,
}

pub struct PathChip<F: PrimeField, H: FieldHasher<F, 2>, const N: usize> {
    siblings: [AssignedCell<F, F>; N],
    direction_bits: [AssignedCell<F, F>; N],
    poseidon_chip: Poseidon2Chip<F, 2>,
    is_eq_chip: IsEqualChip<F>,
    swap_chip: ConditionalSwapChip<F>,
    _hasher: PhantomData<H>,
}

impl<F: PrimeField, H: FieldHasher<F, 2>, const N: usize> PathChip<F, H, N> {
    pub fn configure(meta: &mut ConstraintSystem<F>) -> PathConfig<F, N> {
        let s_path = meta.selector();
        let advices = [(); N].map(|_| meta.advice_column());
        let swap_advices: [Column<Advice>; NUM_OF_SWAP_ADVICE_COLUMNS] =
            [(); NUM_OF_SWAP_ADVICE_COLUMNS].map(|_| meta.advice_column());

        advices
            .iter()
            .for_each(|column| meta.enable_equality(*column));

        // IsEqual reuses the first 4 of the 5 swap columns (gates are selector-gated)
        let is_eq_advices: [Column<Advice>; NUM_OF_UTILITY_ADVICE_COLUMNS] =
            swap_advices[..NUM_OF_UTILITY_ADVICE_COLUMNS].try_into().unwrap();

        PathConfig {
            s_path,
            advices,
            poseidon_config: Poseidon2Chip::<F, 2>::configure(meta),
            is_eq_config: IsEqualChip::configure(meta, is_eq_advices),
            swap_config: ConditionalSwapChip::configure(meta, swap_advices),
        }
    }

    pub fn from_native(
        config: PathConfig<F, N>,
        layouter: &mut impl Layouter<F>,
        native: Path<F, H, N>,
    ) -> Result<Self, Error> {
        let (siblings, direction_bits) = layouter.assign_region(
            || "path",
            |mut region| {
                config.s_path.enable(&mut region, 0)?;

                let siblings = (0..N)
                    .map(|i| {
                        // Extract the sibling node based on direction bit
                        let sibling = if native.direction_bits[i] {
                            native.path[i].0 // we're right child, sibling is the left node
                        } else {
                            native.path[i].1 // we're left child, sibling is the right node
                        };
                        region.assign_advice(
                            || format!("sibling[{}]", i),
                            config.advices[i],
                            0,
                            || Value::known(sibling),
                        )
                    })
                    .collect::<Result<Vec<AssignedCell<F, F>>, Error>>()?;

                let direction_bits = (0..N)
                    .map(|i| {
                        let bit = if native.direction_bits[i] {
                            F::ONE
                        } else {
                            F::ZERO
                        };
                        region.assign_advice(
                            || format!("direction_bit[{}]", i),
                            config.advices[i],
                            1,
                            || Value::known(bit),
                        )
                    })
                    .collect::<Result<Vec<AssignedCell<F, F>>, Error>>()?;

                Ok((
                    siblings.try_into().unwrap(),
                    direction_bits.try_into().unwrap(),
                ))
            },
        )?;

        Ok(PathChip {
            siblings,
            direction_bits,
            poseidon_chip: Poseidon2Chip::<F, 2>::construct(config.poseidon_config),
            is_eq_chip: IsEqualChip::construct(config.is_eq_config, ()),
            swap_chip: ConditionalSwapChip::construct(config.swap_config, ()),
            _hasher: PhantomData,
        })
    }

    pub fn calculate_root(
        &self,
        layouter: &mut impl Layouter<F>,
        leaf: AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, Error> {
        let mut previous_hash = leaf;

        for i in 0..N {
            // Swap (previous_hash, sibling) based on direction bit to get (left, right)
            let (left, right) = self.swap_chip.swap(
                layouter,
                previous_hash,
                self.siblings[i].clone(),
                self.direction_bits[i].clone(),
            )?;
            previous_hash = self.poseidon_chip.hash(layouter, &[left, right])?;
        }

        Ok(previous_hash)
    }

    pub fn check_membership(
        &self,
        layouter: &mut impl Layouter<F>,
        root_hash: AssignedCell<F, F>,
        leaf: AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, Error> {
        let computed_root = self.calculate_root(layouter, leaf)?;

        self.is_eq_chip
            .is_eq_with_output(layouter, computed_root, root_hash)
    }
}

// ========== Sparse Path Chip ==========

#[derive(Clone)]
pub struct SparsePathConfig<F: PrimeField, const MAX_K: usize> {
    s_path: Selector,
    advices: [Column<Advice>; MAX_K],
    poseidon_config: Poseidon2Config<F>,
    swap_config: ConditionalSwapConfig<F>,
    cond_select_config: ConditionalSelectConfig<F>,
    is_eq_config: IsEqualConfig<F>,
}

pub struct SparsePathChip<F: PrimeField, H: FieldHasher<F, 2>, const MAX_K: usize> {
    siblings: [AssignedCell<F, F>; MAX_K],
    direction_bits: [AssignedCell<F, F>; MAX_K],
    is_active: [AssignedCell<F, F>; MAX_K],
    poseidon_chip: Poseidon2Chip<F, 2>,
    swap_chip: ConditionalSwapChip<F>,
    cond_select_chip: ConditionalSelectChip<F>,
    is_eq_chip: IsEqualChip<F>,
    _hasher: PhantomData<H>,
}

impl<F: PrimeField, H: FieldHasher<F, 2>, const MAX_K: usize>
    SparsePathChip<F, H, MAX_K>
{
    pub fn configure(meta: &mut ConstraintSystem<F>) -> SparsePathConfig<F, MAX_K> {
        let s_path = meta.selector();
        let advices = [(); MAX_K].map(|_| meta.advice_column());
        let swap_advices: [Column<Advice>; NUM_OF_SWAP_ADVICE_COLUMNS] =
            [(); NUM_OF_SWAP_ADVICE_COLUMNS].map(|_| meta.advice_column());

        advices
            .iter()
            .for_each(|column| meta.enable_equality(*column));

        // ConditionalSelect and IsEqual reuse the first 4 of the 5 swap columns
        // (gates are selector-gated so they don't conflict)
        let utility_advices: [Column<Advice>; NUM_OF_UTILITY_ADVICE_COLUMNS] =
            swap_advices[..NUM_OF_UTILITY_ADVICE_COLUMNS]
                .try_into()
                .unwrap();

        SparsePathConfig {
            s_path,
            advices,
            poseidon_config: Poseidon2Chip::<F, 2>::configure(meta),
            swap_config: ConditionalSwapChip::configure(meta, swap_advices),
            cond_select_config: ConditionalSelectChip::configure(meta, utility_advices),
            is_eq_config: IsEqualChip::configure(meta, utility_advices),
        }
    }

    pub fn from_native(
        config: SparsePathConfig<F, MAX_K>,
        layouter: &mut impl Layouter<F>,
        sparse_path: SparsePath<F, H>,
    ) -> Result<Self, Error> {
        let (padded_siblings, padded_dir_bits, padded_active) =
            sparse_path.to_padded_arrays::<MAX_K>();

        let (siblings, direction_bits, is_active) = layouter.assign_region(
            || "sparse_path",
            |mut region| {
                config.s_path.enable(&mut region, 0)?;

                let siblings = (0..MAX_K)
                    .map(|i| {
                        region.assign_advice(
                            || format!("sibling[{}]", i),
                            config.advices[i],
                            0,
                            || Value::known(padded_siblings[i]),
                        )
                    })
                    .collect::<Result<Vec<AssignedCell<F, F>>, Error>>()?;

                let direction_bits = (0..MAX_K)
                    .map(|i| {
                        let bit = if padded_dir_bits[i] { F::ONE } else { F::ZERO };
                        region.assign_advice(
                            || format!("direction_bit[{}]", i),
                            config.advices[i],
                            1,
                            || Value::known(bit),
                        )
                    })
                    .collect::<Result<Vec<AssignedCell<F, F>>, Error>>()?;

                let is_active = (0..MAX_K)
                    .map(|i| {
                        let active = if padded_active[i] { F::ONE } else { F::ZERO };
                        region.assign_advice(
                            || format!("is_active[{}]", i),
                            config.advices[i],
                            2,
                            || Value::known(active),
                        )
                    })
                    .collect::<Result<Vec<AssignedCell<F, F>>, Error>>()?;

                Ok((
                    siblings.try_into().unwrap(),
                    direction_bits.try_into().unwrap(),
                    is_active.try_into().unwrap(),
                ))
            },
        )?;

        Ok(SparsePathChip {
            siblings,
            direction_bits,
            is_active,
            poseidon_chip: Poseidon2Chip::construct(config.poseidon_config),
            swap_chip: ConditionalSwapChip::construct(config.swap_config, ()),
            cond_select_chip: ConditionalSelectChip::construct(config.cond_select_config, ()),
            is_eq_chip: IsEqualChip::construct(config.is_eq_config, ()),
            _hasher: PhantomData,
        })
    }

    pub fn calculate_root(
        &self,
        layouter: &mut impl Layouter<F>,
        leaf: AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, Error> {
        let mut prev = leaf;

        for i in 0..MAX_K {
            // Step 1: Swap based on direction bit (1 row)
            let (left, right) = self.swap_chip.swap(
                layouter,
                prev.clone(),
                self.siblings[i].clone(),
                self.direction_bits[i].clone(),
            )?;
            // Step 2: Poseidon hash (~64 rows)
            let hash_result = self.poseidon_chip.hash(layouter, &[left, right])?;
            // Step 3: Conditional select -- active: use hash, inactive: pass through (1 row)
            prev = self.cond_select_chip.conditional_select(
                layouter,
                hash_result,
                prev,
                self.is_active[i].clone(),
            )?;
        }

        Ok(prev)
    }

    pub fn check_membership(
        &self,
        layouter: &mut impl Layouter<F>,
        compact_root: AssignedCell<F, F>,
        leaf: AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, Error> {
        let computed_root = self.calculate_root(layouter, leaf)?;

        self.is_eq_chip
            .is_eq_with_output(layouter, computed_root, compact_root)
    }
}

#[cfg(test)]
mod test {

    use super::{PathChip, PathConfig, SparsePathChip, SparsePathConfig};
    use crate::measure;
    use crate::utilities::{AssertEqualChip, AssertEqualConfig};
    use ff::{Field, FromUniformBytes, PrimeField};
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, SingleVerifier};
    use halo2_proofs::poly::commitment::Params;
    use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{Advice, Circuit, Column, ConstraintSystem, Error},
    };
    use pasta_curves::{EqAffine, Fp};
    use rand::rngs::OsRng;
    use smt::poseidon::FieldHasher;
    use smt::poseidon2::Poseidon2;
    use smt::smt::SparseMerkleTree;
    use std::clone::Clone;
    use std::marker::PhantomData;
    use std::time::Instant;

    #[derive(Clone)]
    struct TestConfig<F: PrimeField, H: FieldHasher<F, 2>, const N: usize> {
        path_config: PathConfig<F, N>,
        advices: [Column<Advice>; 3],
        assert_equal_config: AssertEqualConfig<F>,
        _hasher: PhantomData<H>,
    }

    struct TestCircuit<F: PrimeField, H: FieldHasher<F, 2>, const N: usize> {
        leaves: [F; 3],
        empty_leaf: [u8; 64],
        hasher: H,
    }

    impl<
            F: PrimeField + FromUniformBytes<64> + Ord,
            H: FieldHasher<F, 2> + Clone,
            const N: usize,
        > Circuit<F> for TestCircuit<F, H, N>
    {
        type Config = TestConfig<F, H, N>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                leaves: [F::ZERO, F::ZERO, F::ZERO],
                empty_leaf: [0u8; 64],
                hasher: H::hasher(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = [(); 3].map(|_| meta.advice_column());
            advices
                .iter()
                .for_each(|column| meta.enable_equality(*column));

            TestConfig {
                path_config: PathChip::<F, H, N>::configure(meta),
                advices,
                assert_equal_config: AssertEqualChip::configure(meta, [advices[0], advices[1]]),
                _hasher: PhantomData,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let smt = SparseMerkleTree::<F, H, N>::new_sequential(
                &self.leaves,
                &self.hasher.clone(),
                &self.empty_leaf,
            )
            .unwrap();
            let path = smt.generate_membership_proof(0);
            let root = path
                .calculate_root(&self.leaves[0], &self.hasher.clone())
                .unwrap();

            let (root_cell, leaf_cell, one) = layouter.assign_region(
                || "test circuit",
                |mut region| {
                    let root_cell = region.assign_advice(
                        || "root",
                        config.advices[0],
                        0,
                        || Value::known(root),
                    )?;

                    let leaf_cell = region.assign_advice(
                        || "leaf",
                        config.advices[1],
                        0,
                        || Value::known(self.leaves[0]),
                    )?;

                    let one = region.assign_advice(
                        || "one",
                        config.advices[2],
                        0,
                        || Value::known(F::ONE),
                    )?;
                    Ok((root_cell, leaf_cell, one))
                },
            )?;

            let path_chip = PathChip::<F, H, N>::from_native(
                config.path_config,
                &mut layouter,
                path,
            )?;
            let res = path_chip.check_membership(&mut layouter, root_cell, leaf_cell)?;

            let assert_equal_chip = AssertEqualChip::construct(config.assert_equal_config, ());
            assert_equal_chip.assert_equal(&mut layouter, res, one)?;

            Ok(())
        }
    }

    #[test]
    fn should_verify_path() {
        // Circuit is very small, we pick a small value here
        let k = 17;

        let empty_leaf = [0u8; 64];
        let rng = OsRng;
        let leaves = [Fp::random(rng), Fp::random(rng), Fp::random(rng)];
        const HEIGHT: usize = 20;

        let circuit = TestCircuit::<Fp, Poseidon2<Fp, 2>, HEIGHT> {
            leaves,
            empty_leaf,
            hasher: Poseidon2::<Fp, 2>::new(),
        };

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));

        let now = Instant::now();
        let params: Params<EqAffine> = Params::new(k);
        println!("Params::new(k={}) time: {:?}", k, now.elapsed());

        let mut params_buf = vec![];
        params.write(&mut params_buf).expect("params serialization should not fail");
        println!("Params size: {:.2} MB", params_buf.len() as f64 / (1024.0 * 1024.0));

        let now = Instant::now();
        let vk = keygen_vk(&params, &circuit).expect("keygen_vk should not fail");
        println!("keygen_vk time: {:?}", now.elapsed());

        let now = Instant::now();
        let pk = keygen_pk(&params, vk, &circuit).expect("keygen_pk should not fail");
        println!("keygen_pk time: {:?}", now.elapsed());

        let now = Instant::now();
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        create_proof(&params, &pk, &[circuit], &[&[]], OsRng, &mut transcript)
            .expect("proof generation should not fail");
        let proof: Vec<u8> = transcript.finalize();
        println!("create_proof time is {:?}", now.elapsed());

        let now = Instant::now();
        let strategy = SingleVerifier::new(&params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        let result = verify_proof(&params, pk.get_vk(), strategy, &[&[]], &mut transcript);
        println!("verify_proof time is {:?}", now.elapsed());

        assert!(result.is_ok());
    }

    #[test]
    #[allow(path_statements)]
    fn should_verify_path_benchmark() {
        // Circuit is very small, we pick a small value here
        let k = 13;

        let empty_leaf = [0u8; 64];
        let rng = OsRng;
        let leaves = [Fp::random(rng), Fp::random(rng), Fp::random(rng)];
        const HEIGHT: usize = 3;
        let num_iter = 3;

        measure!(
            {
                let circuit = TestCircuit::<Fp, Poseidon2<Fp, 2>, HEIGHT> {
                    leaves,
                    empty_leaf,
                    hasher: Poseidon2::<Fp, 2>::new(),
                };

                let prover = MockProver::run(k, &circuit, vec![]).unwrap();
                assert_eq!(prover.verify(), Ok(()));

                let params: Params<EqAffine> = Params::new(k);
                let vk = keygen_vk(&params, &circuit).expect("keygen_vk should not fail");
                let pk = keygen_pk(&params, vk, &circuit).expect("keygen_pk should not fail");
                let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
                create_proof(&params, &pk, &[circuit], &[&[]], OsRng, &mut transcript)
                    .expect("proof generation should not fail");
                let proof: Vec<u8> = transcript.finalize();

                let strategy = SingleVerifier::new(&params);
                let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
                let result = verify_proof(&params, pk.get_vk(), strategy, &[&[]], &mut transcript);
                assert!(result.is_ok());
            },
            "hola2",
            "proof",
            num_iter
        );
    }

    // ========== Sparse Path Circuit Tests ==========

    #[derive(Clone)]
    struct SparseTestConfig<F: PrimeField, H: FieldHasher<F, 2>, const MAX_K: usize> {
        sparse_path_config: SparsePathConfig<F, MAX_K>,
        advices: [Column<Advice>; 3],
        assert_equal_config: AssertEqualConfig<F>,
        _hasher: PhantomData<H>,
    }

    struct SparseTestCircuit<
        F: PrimeField,
        H: FieldHasher<F, 2>,
        const N: usize,
        const MAX_K: usize,
    > {
        leaves: [F; 3],
        empty_leaf: [u8; 64],
        hasher: H,
    }

    impl<
            F: PrimeField + FromUniformBytes<64> + Ord,
            H: FieldHasher<F, 2> + Clone,
            const N: usize,
            const MAX_K: usize,
        > Circuit<F> for SparseTestCircuit<F, H, N, MAX_K>
    {
        type Config = SparseTestConfig<F, H, MAX_K>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                leaves: [F::ZERO; 3],
                empty_leaf: [0u8; 64],
                hasher: H::hasher(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = [(); 3].map(|_| meta.advice_column());
            advices
                .iter()
                .for_each(|column| meta.enable_equality(*column));

            SparseTestConfig {
                sparse_path_config: SparsePathChip::<F, H, MAX_K>::configure(meta),
                advices,
                assert_equal_config: AssertEqualChip::configure(meta, [advices[0], advices[1]]),
                _hasher: PhantomData,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let smt = SparseMerkleTree::<F, H, N>::new_sequential(
                &self.leaves,
                &self.hasher.clone(),
                &self.empty_leaf,
            )
            .unwrap();

            let sparse_path = smt.generate_sparse_membership_proof(0);
            let compact_root = sparse_path
                .calculate_compact_root(&self.leaves[0], &self.hasher.clone())
                .unwrap();

            let (compact_root_cell, leaf_cell, one) = layouter.assign_region(
                || "test circuit",
                |mut region| {
                    let root_cell = region.assign_advice(
                        || "compact_root",
                        config.advices[0],
                        0,
                        || Value::known(compact_root),
                    )?;
                    let leaf_cell = region.assign_advice(
                        || "leaf",
                        config.advices[1],
                        0,
                        || Value::known(self.leaves[0]),
                    )?;
                    let one = region.assign_advice(
                        || "one",
                        config.advices[2],
                        0,
                        || Value::known(F::ONE),
                    )?;
                    Ok((root_cell, leaf_cell, one))
                },
            )?;

            let sparse_path_chip = SparsePathChip::<F, H, MAX_K>::from_native(
                config.sparse_path_config,
                &mut layouter,
                sparse_path.clone(),
            )?;
            let res =
                sparse_path_chip.check_membership(&mut layouter, compact_root_cell, leaf_cell)?;

            let assert_equal_chip = AssertEqualChip::construct(config.assert_equal_config, ());
            assert_equal_chip.assert_equal(&mut layouter, res, one)?;

            // Verify gap levels natively (outside the circuit)
            let full_root = sparse_path
                .calculate_full_root(&self.leaves[0], &self.hasher.clone(), 0)
                .unwrap();
            assert_eq!(full_root, smt.root(), "Gap verification failed");

            Ok(())
        }
    }

    #[test]
    fn test_sparse_path_circuit() {
        let k = 13;

        let empty_leaf = [0u8; 64];
        let rng = OsRng;
        let leaves = [Fp::random(rng), Fp::random(rng), Fp::random(rng)];
        const HEIGHT: usize = 10;
        const MAX_K: usize = 4;

        let circuit = SparseTestCircuit::<Fp, Poseidon2<Fp, 2>, HEIGHT, MAX_K> {
            leaves,
            empty_leaf,
            hasher: Poseidon2::<Fp, 2>::new(),
        };

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn test_sparse_path_full_prove() {
        let k = 13;

        let empty_leaf = [0u8; 64];
        let rng = OsRng;
        let leaves = [Fp::random(rng), Fp::random(rng), Fp::random(rng)];
        const HEIGHT: usize = 20;
        const MAX_K: usize = 8;

        let circuit = SparseTestCircuit::<Fp, Poseidon2<Fp, 2>, HEIGHT, MAX_K> {
            leaves,
            empty_leaf,
            hasher: Poseidon2::<Fp, 2>::new(),
        };

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));

        let now = Instant::now();
        let params: Params<EqAffine> = Params::new(k);
        println!("Sparse: Params::new(k={}) time: {:?}", k, now.elapsed());

        let mut params_buf = vec![];
        params
            .write(&mut params_buf)
            .expect("params serialization should not fail");
        println!(
            "Sparse: Params size: {:.2} MB",
            params_buf.len() as f64 / (1024.0 * 1024.0)
        );

        let now = Instant::now();
        let vk = keygen_vk(&params, &circuit).expect("keygen_vk should not fail");
        println!("Sparse: keygen_vk time: {:?}", now.elapsed());

        let now = Instant::now();
        let pk = keygen_pk(&params, vk, &circuit).expect("keygen_pk should not fail");
        println!("Sparse: keygen_pk time: {:?}", now.elapsed());

        let now = Instant::now();
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        create_proof(&params, &pk, &[circuit], &[&[]], OsRng, &mut transcript)
            .expect("proof generation should not fail");
        let proof: Vec<u8> = transcript.finalize();
        println!("Sparse: create_proof time is {:?}", now.elapsed());

        let now = Instant::now();
        let strategy = SingleVerifier::new(&params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        let result = verify_proof(&params, pk.get_vk(), strategy, &[&[]], &mut transcript);
        println!("Sparse: verify_proof time is {:?}", now.elapsed());

        assert!(result.is_ok());
    }

    #[test]
    fn test_sparse_vs_dense_consistency() {
        let empty_leaf = [0u8; 64];
        let rng = OsRng;
        let leaves = [Fp::random(rng), Fp::random(rng), Fp::random(rng)];
        const HEIGHT: usize = 10;
        let hasher = Poseidon2::<Fp, 2>::new();

        let smt = SparseMerkleTree::<Fp, Poseidon2<Fp, 2>, HEIGHT>::new_sequential(
            &leaves,
            &hasher,
            &empty_leaf,
        )
        .unwrap();

        // Dense path: standard membership proof
        let dense_path = smt.generate_membership_proof(0);
        let dense_root = dense_path
            .calculate_root(&leaves[0], &hasher)
            .unwrap();

        // Sparse path: compact root + gap verification = standard root
        let sparse_path = smt.generate_sparse_membership_proof(0);
        let full_root = sparse_path
            .calculate_full_root(&leaves[0], &hasher, 0)
            .unwrap();

        // Both should produce the same standard root
        assert_eq!(
            dense_root, full_root,
            "Dense and sparse paths produce different roots"
        );
        assert_eq!(dense_root, smt.root(), "Root doesn't match SMT root");
    }

    // ========== N=53 Realistic Benchmarks ==========

    #[derive(Clone)]
    struct DenseBenchConfig<F: PrimeField, H: FieldHasher<F, 2>, const N: usize> {
        path_config: PathConfig<F, N>,
        advices: [Column<Advice>; 3],
        assert_equal_config: AssertEqualConfig<F>,
        _hasher: PhantomData<H>,
    }

    struct DenseBenchCircuit<F: PrimeField, H: FieldHasher<F, 2>, const N: usize> {
        root: F,
        leaf: F,
        path: smt::smt::Path<F, H, N>,
    }

    impl<
            F: PrimeField + FromUniformBytes<64> + Ord,
            H: FieldHasher<F, 2> + Clone,
            const N: usize,
        > Circuit<F> for DenseBenchCircuit<F, H, N>
    {
        type Config = DenseBenchConfig<F, H, N>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                root: F::ZERO,
                leaf: F::ZERO,
                path: smt::smt::Path {
                    path: [(F::ZERO, F::ZERO); N],
                    direction_bits: [false; N],
                    marker: PhantomData,
                },
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = [(); 3].map(|_| meta.advice_column());
            advices
                .iter()
                .for_each(|column| meta.enable_equality(*column));

            DenseBenchConfig {
                path_config: PathChip::<F, H, N>::configure(meta),
                advices,
                assert_equal_config: AssertEqualChip::configure(meta, [advices[0], advices[1]]),
                _hasher: PhantomData,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let (root_cell, leaf_cell, one) = layouter.assign_region(
                || "bench circuit",
                |mut region| {
                    let root_cell = region.assign_advice(
                        || "root",
                        config.advices[0],
                        0,
                        || Value::known(self.root),
                    )?;
                    let leaf_cell = region.assign_advice(
                        || "leaf",
                        config.advices[1],
                        0,
                        || Value::known(self.leaf),
                    )?;
                    let one = region.assign_advice(
                        || "one",
                        config.advices[2],
                        0,
                        || Value::known(F::ONE),
                    )?;
                    Ok((root_cell, leaf_cell, one))
                },
            )?;

            let path_chip = PathChip::<F, H, N>::from_native(
                config.path_config,
                &mut layouter,
                self.path.clone(),
            )?;
            let res = path_chip.check_membership(&mut layouter, root_cell, leaf_cell)?;

            let assert_equal_chip = AssertEqualChip::construct(config.assert_equal_config, ());
            assert_equal_chip.assert_equal(&mut layouter, res, one)?;
            Ok(())
        }
    }

    #[derive(Clone)]
    struct SparseBenchConfig<F: PrimeField, H: FieldHasher<F, 2>, const MAX_K: usize> {
        sparse_path_config: SparsePathConfig<F, MAX_K>,
        advices: [Column<Advice>; 3],
        assert_equal_config: AssertEqualConfig<F>,
        _hasher: PhantomData<H>,
    }

    struct SparseBenchCircuit<F: PrimeField, H: FieldHasher<F, 2>, const MAX_K: usize> {
        compact_root: F,
        leaf: F,
        sparse_path: smt::smt::SparsePath<F, H>,
    }

    impl<
            F: PrimeField + FromUniformBytes<64> + Ord,
            H: FieldHasher<F, 2> + Clone,
            const MAX_K: usize,
        > Circuit<F> for SparseBenchCircuit<F, H, MAX_K>
    {
        type Config = SparseBenchConfig<F, H, MAX_K>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                compact_root: F::ZERO,
                leaf: F::ZERO,
                sparse_path: smt::smt::SparsePath {
                    entries: Vec::new(),
                    tree_height: 0,
                    empty_hashes: Vec::new(),
                    marker: PhantomData,
                },
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = [(); 3].map(|_| meta.advice_column());
            advices
                .iter()
                .for_each(|column| meta.enable_equality(*column));

            SparseBenchConfig {
                sparse_path_config: SparsePathChip::<F, H, MAX_K>::configure(meta),
                advices,
                assert_equal_config: AssertEqualChip::configure(meta, [advices[0], advices[1]]),
                _hasher: PhantomData,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let (root_cell, leaf_cell, one) = layouter.assign_region(
                || "bench circuit",
                |mut region| {
                    let root_cell = region.assign_advice(
                        || "compact_root",
                        config.advices[0],
                        0,
                        || Value::known(self.compact_root),
                    )?;
                    let leaf_cell = region.assign_advice(
                        || "leaf",
                        config.advices[1],
                        0,
                        || Value::known(self.leaf),
                    )?;
                    let one = region.assign_advice(
                        || "one",
                        config.advices[2],
                        0,
                        || Value::known(F::ONE),
                    )?;
                    Ok((root_cell, leaf_cell, one))
                },
            )?;

            let sparse_path_chip = SparsePathChip::<F, H, MAX_K>::from_native(
                config.sparse_path_config,
                &mut layouter,
                self.sparse_path.clone(),
            )?;
            let res =
                sparse_path_chip.check_membership(&mut layouter, root_cell, leaf_cell)?;

            let assert_equal_chip = AssertEqualChip::construct(config.assert_equal_config, ());
            assert_equal_chip.assert_equal(&mut layouter, res, one)?;
            Ok(())
        }
    }

    #[test]
    fn bench_n53_dense_vs_sparse() {
        use std::collections::BTreeMap;

        let rng = OsRng;
        let empty_leaf = [0u8; 64];
        let hasher = Poseidon2::<Fp, 2>::new();
        const HEIGHT: usize = 53;
        const MAX_K: usize = 32;

        // Insert leaves at positions 0, 1, 2, 4, 8, ..., 2^26
        // Each power-of-2 position forces a non-empty sibling at a different
        // level, simulating the sparsity of ~100M items (2^27).
        let mut leaf_map = BTreeMap::new();
        let leaf0 = Fp::random(rng);
        leaf_map.insert(0u32, leaf0);
        for i in 0..27u32 {
            leaf_map.insert(1u32 << i, Fp::random(rng));
        }

        println!("Building SMT with HEIGHT={}, {} leaves...", HEIGHT, leaf_map.len());
        let now = Instant::now();
        let smt = SparseMerkleTree::<Fp, Poseidon2<Fp, 2>, HEIGHT>::new(
            &leaf_map,
            &hasher,
            &empty_leaf,
        )
        .unwrap();
        println!("SMT built in {:?}", now.elapsed());

        let sparse_proof = smt.generate_sparse_membership_proof(0);
        println!(
            "Non-zero siblings: K = {} out of N = {} levels\n",
            sparse_proof.entries.len(),
            HEIGHT
        );

        let dense_path = smt.generate_membership_proof(0);
        let dense_root = dense_path.calculate_root(&leaf0, &hasher).unwrap();
        assert_eq!(dense_root, smt.root());

        let compact_root = sparse_proof
            .calculate_compact_root(&leaf0, &hasher)
            .unwrap();
        assert!(sparse_proof
            .verify_against_root(&leaf0, &hasher, 0, &smt.root())
            .unwrap());

        // ===== Dense Benchmark =====
        println!("===== Dense PathChip (N={}) =====", HEIGHT);

        let dense_circuit = DenseBenchCircuit::<Fp, Poseidon2<Fp, 2>, HEIGHT> {
            root: dense_root,
            leaf: leaf0,
            path: dense_path,
        };

        let k_dense = 12;
        let now = Instant::now();
        let prover = MockProver::run(k_dense, &dense_circuit, vec![]).unwrap();
        println!("Dense MockProver (k={}) time: {:?}", k_dense, now.elapsed());
        assert_eq!(prover.verify(), Ok(()));

        let now = Instant::now();
        let params_dense: Params<EqAffine> = Params::new(k_dense);
        println!("Dense Params::new(k={}) time: {:?}", k_dense, now.elapsed());

        let mut params_buf = vec![];
        params_dense.write(&mut params_buf).unwrap();
        println!("Dense Params size: {:.2} MB", params_buf.len() as f64 / (1024.0 * 1024.0));

        let now = Instant::now();
        let vk = keygen_vk(&params_dense, &dense_circuit).expect("keygen_vk should not fail");
        println!("Dense keygen_vk time: {:?}", now.elapsed());

        let now = Instant::now();
        let pk = keygen_pk(&params_dense, vk, &dense_circuit).expect("keygen_pk should not fail");
        println!("Dense keygen_pk time: {:?}", now.elapsed());

        let now = Instant::now();
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        create_proof(&params_dense, &pk, &[dense_circuit], &[&[]], OsRng, &mut transcript)
            .expect("proof generation should not fail");
        let proof = transcript.finalize();
        println!("Dense create_proof time: {:?}", now.elapsed());

        let now = Instant::now();
        let strategy = SingleVerifier::new(&params_dense);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        let result = verify_proof(&params_dense, pk.get_vk(), strategy, &[&[]], &mut transcript);
        println!("Dense verify_proof time: {:?}", now.elapsed());
        assert!(result.is_ok());

        // ===== Sparse Benchmark =====
        println!("\n===== Sparse SparsePathChip (MAX_K={}) =====", MAX_K);

        let sparse_circuit = SparseBenchCircuit::<Fp, Poseidon2<Fp, 2>, MAX_K> {
            compact_root,
            leaf: leaf0,
            sparse_path: sparse_proof,
        };

        let k_sparse = 12;
        let now = Instant::now();
        let prover = MockProver::run(k_sparse, &sparse_circuit, vec![]).unwrap();
        println!(
            "Sparse MockProver (k={}) time: {:?}",
            k_sparse,
            now.elapsed()
        );
        assert_eq!(prover.verify(), Ok(()));

        let now = Instant::now();
        let params_sparse: Params<EqAffine> = Params::new(k_sparse);
        println!(
            "Sparse Params::new(k={}) time: {:?}",
            k_sparse,
            now.elapsed()
        );

        let mut params_buf = vec![];
        params_sparse.write(&mut params_buf).unwrap();
        println!(
            "Sparse Params size: {:.2} MB",
            params_buf.len() as f64 / (1024.0 * 1024.0)
        );

        let now = Instant::now();
        let vk =
            keygen_vk(&params_sparse, &sparse_circuit).expect("keygen_vk should not fail");
        println!("Sparse keygen_vk time: {:?}", now.elapsed());

        let now = Instant::now();
        let pk =
            keygen_pk(&params_sparse, vk, &sparse_circuit).expect("keygen_pk should not fail");
        println!("Sparse keygen_pk time: {:?}", now.elapsed());

        let now = Instant::now();
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        create_proof(
            &params_sparse,
            &pk,
            &[sparse_circuit],
            &[&[]],
            OsRng,
            &mut transcript,
        )
        .expect("proof generation should not fail");
        let proof = transcript.finalize();
        println!("Sparse create_proof time: {:?}", now.elapsed());

        let now = Instant::now();
        let strategy = SingleVerifier::new(&params_sparse);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        let result =
            verify_proof(&params_sparse, pk.get_vk(), strategy, &[&[]], &mut transcript);
        println!("Sparse verify_proof time: {:?}", now.elapsed());
        assert!(result.is_ok());
    }
}
