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
    ConditionalSwapChip, ConditionalSwapConfig,
    IsEqualChip, IsEqualConfig, NUM_OF_SWAP_ADVICE_COLUMNS, NUM_OF_UTILITY_ADVICE_COLUMNS,
};
use ff::PrimeField;
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error, Selector},
};
use smt::poseidon2::FieldHasher;
use smt::smt::Path;
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

#[cfg(test)]
mod test {

    use super::{PathChip, PathConfig};
    use crate::measure;
    use crate::poseidon2_chip::{Poseidon2Chip, Poseidon2Config};
    use crate::utilities::{AssertEqualChip, AssertEqualConfig};
    use ff::{Field, FromUniformBytes, PrimeField};
    use halo2_proofs::circuit::AssignedCell;
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
    use smt::poseidon2::FieldHasher;
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

    #[test]
    fn bench_n53_dense() {
        use std::collections::BTreeMap;

        let rng = OsRng;
        let empty_leaf = [0u8; 64];
        let hasher = Poseidon2::<Fp, 2>::new();
        const HEIGHT: usize = 53;

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

        let dense_path = smt.generate_membership_proof(0);
        let dense_root = dense_path.calculate_root(&leaf0, &hasher).unwrap();
        assert_eq!(dense_root, smt.root());

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
    }

    // ========== Poseidon1 vs Poseidon2 Benchmark Circuits ==========

    // ---- Poseidon1 hash chain circuit (uses Poseidon2Chip) ----

    #[derive(Clone)]
    struct P1HashChainConfig<F: PrimeField> {
        poseidon_config: Poseidon2Config<F>,
        input: [Column<Advice>; 2],
        output: Column<Advice>,
    }

    struct P1HashChainCircuit<F: PrimeField> {
        pairs: Vec<[F; 2]>,
        expected_final: F,
    }

    impl<F: PrimeField + FromUniformBytes<64> + Ord> Circuit<F> for P1HashChainCircuit<F> {
        type Config = P1HashChainConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                pairs: vec![[F::ZERO; 2]; self.pairs.len()],
                expected_final: F::ZERO,
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let input = [meta.advice_column(), meta.advice_column()];
            let output = meta.advice_column();
            input.iter().for_each(|c| meta.enable_equality(*c));
            meta.enable_equality(output);

            P1HashChainConfig {
                poseidon_config: Poseidon2Chip::<F, 2>::configure(meta),
                input,
                output,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let mut last_hash: Option<AssignedCell<F, F>> = None;
            let num_hashes = self.pairs.len();

            for i in 0..num_hashes {
                let inputs = layouter.assign_region(
                    || format!("p1_inputs_{}", i),
                    |mut region| {
                        let a = region.assign_advice(
                            || "a",
                            config.input[0],
                            0,
                            || Value::known(self.pairs[i][0]),
                        )?;
                        let b = region.assign_advice(
                            || "b",
                            config.input[1],
                            0,
                            || Value::known(self.pairs[i][1]),
                        )?;
                        Ok([a, b])
                    },
                )?;

                let chip =
                    Poseidon2Chip::<F, 2>::construct(config.poseidon_config.clone());
                last_hash = Some(chip.hash(
                    &mut layouter.namespace(|| format!("p1_hash_{}", i)),
                    &inputs,
                )?);
            }

            layouter.assign_region(
                || "p1_check_output",
                |mut region| {
                    let expected = region.assign_advice(
                        || "expected",
                        config.output,
                        0,
                        || Value::known(self.expected_final),
                    )?;
                    region.constrain_equal(
                        last_hash.as_ref().unwrap().cell(),
                        expected.cell(),
                    )
                },
            )
        }
    }

    // ---- Poseidon2 hash chain circuit (uses batched Poseidon2Chip) ----

    #[derive(Clone)]
    struct P2HashChainConfig<F: PrimeField> {
        poseidon2_config: Poseidon2Config<F>,
        input: [Column<Advice>; 2],
        output: Column<Advice>,
    }

    struct P2HashChainCircuit<F: PrimeField> {
        pairs: Vec<[F; 2]>,
        expected_final: F,
    }

    impl<F: PrimeField + FromUniformBytes<64> + Ord> Circuit<F> for P2HashChainCircuit<F> {
        type Config = P2HashChainConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                pairs: vec![[F::ZERO; 2]; self.pairs.len()],
                expected_final: F::ZERO,
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let input = [meta.advice_column(), meta.advice_column()];
            let output = meta.advice_column();
            input.iter().for_each(|c| meta.enable_equality(*c));
            meta.enable_equality(output);

            P2HashChainConfig {
                poseidon2_config: Poseidon2Chip::<F, 2>::configure(meta),
                input,
                output,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let mut last_hash: Option<AssignedCell<F, F>> = None;
            let num_hashes = self.pairs.len();

            for i in 0..num_hashes {
                let inputs = layouter.assign_region(
                    || format!("p2_inputs_{}", i),
                    |mut region| {
                        let a = region.assign_advice(
                            || "a",
                            config.input[0],
                            0,
                            || Value::known(self.pairs[i][0]),
                        )?;
                        let b = region.assign_advice(
                            || "b",
                            config.input[1],
                            0,
                            || Value::known(self.pairs[i][1]),
                        )?;
                        Ok([a, b])
                    },
                )?;

                let chip =
                    Poseidon2Chip::<F, 2>::construct(config.poseidon2_config.clone());
                last_hash = Some(chip.hash(
                    &mut layouter.namespace(|| format!("p2_hash_{}", i)),
                    &inputs,
                )?);
            }

            layouter.assign_region(
                || "p2_check_output",
                |mut region| {
                    let expected = region.assign_advice(
                        || "expected",
                        config.output,
                        0,
                        || Value::known(self.expected_final),
                    )?;
                    region.constrain_equal(
                        last_hash.as_ref().unwrap().cell(),
                        expected.cell(),
                    )
                },
            )
        }
    }

    // ---- Benchmark: Poseidon1 vs Poseidon2-batched ----

    #[test]
    #[ignore]
    fn bench_poseidon1_vs_poseidon2() {
        const HEIGHT: usize = 20;
        let k = 13u32;
        let rng = OsRng;

        // Generate random pairs for the hash chain
        let pairs: Vec<[Fp; 2]> = (0..HEIGHT)
            .map(|_| [Fp::random(rng), Fp::random(rng)])
            .collect();

        // Compute expected final hash for each variant (last pair only)
        let p1_hasher = Poseidon2::<Fp, 2>::new();
        let p1_expected = p1_hasher.hash(pairs[HEIGHT - 1]).unwrap();

        let p2_hasher = Poseidon2::<Fp, 2>::new();
        let p2_expected = p2_hasher.hash(pairs[HEIGHT - 1]).unwrap();

        // ===== Part 1: Native Hash Throughput =====
        let iterations = 10_000usize;
        let bench_inputs: Vec<[Fp; 2]> = (0..iterations)
            .map(|_| [Fp::random(rng), Fp::random(rng)])
            .collect();

        let start = Instant::now();
        for inp in &bench_inputs {
            let _ = p1_hasher.hash(*inp).unwrap();
        }
        let p1_native = start.elapsed();

        let start = Instant::now();
        for inp in &bench_inputs {
            let _ = p2_hasher.hash(*inp).unwrap();
        }
        let p2_native = start.elapsed();

        println!("\n========================================");
        println!("  NATIVE HASH THROUGHPUT ({} iterations)", iterations);
        println!("========================================");
        println!(
            "  Poseidon1: {:>8.1} ms  ({:.0} hashes/sec)",
            p1_native.as_secs_f64() * 1000.0,
            iterations as f64 / p1_native.as_secs_f64()
        );
        println!(
            "  Poseidon2: {:>8.1} ms  ({:.0} hashes/sec)",
            p2_native.as_secs_f64() * 1000.0,
            iterations as f64 / p2_native.as_secs_f64()
        );

        // ===== Part 2a: Poseidon1 Circuit Proof =====
        let p1_circuit = P1HashChainCircuit::<Fp> {
            pairs: pairs.clone(),
            expected_final: p1_expected,
        };

        println!("\n========================================");
        println!(
            "  POSEIDON1 CIRCUIT (k={}, {} hashes)",
            k, HEIGHT
        );
        println!("========================================");

        let start = Instant::now();
        let prover = MockProver::run(k, &p1_circuit, vec![]).unwrap();
        println!("  MockProver:   {:?}", start.elapsed());
        assert_eq!(prover.verify(), Ok(()));

        let params: Params<EqAffine> = Params::new(k);

        let start = Instant::now();
        let vk = keygen_vk(&params, &p1_circuit).unwrap();
        let t_vk = start.elapsed();
        let start = Instant::now();
        let pk = keygen_pk(&params, vk, &p1_circuit).unwrap();
        let t_pk = start.elapsed();
        println!("  keygen_vk:    {:?}", t_vk);
        println!("  keygen_pk:    {:?}", t_pk);

        let start = Instant::now();
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        create_proof(
            &params,
            &pk,
            &[p1_circuit],
            &[&[]],
            OsRng,
            &mut transcript,
        )
        .unwrap();
        let p1_proof = transcript.finalize();
        println!("  create_proof: {:?}", start.elapsed());
        println!("  proof size:   {} bytes", p1_proof.len());

        let start = Instant::now();
        let strategy = SingleVerifier::new(&params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&p1_proof[..]);
        verify_proof(&params, pk.get_vk(), strategy, &[&[]], &mut transcript).unwrap();
        println!("  verify_proof: {:?}", start.elapsed());

        // ===== Part 2b: Poseidon2-batched Circuit Proof =====
        let p2_circuit = P2HashChainCircuit::<Fp> {
            pairs: pairs.clone(),
            expected_final: p2_expected,
        };

        println!("\n========================================");
        println!(
            "  POSEIDON2-BATCHED CIRCUIT (k={}, {} hashes)",
            k, HEIGHT
        );
        println!("========================================");

        let start = Instant::now();
        let prover = MockProver::run(k, &p2_circuit, vec![]).unwrap();
        println!("  MockProver:   {:?}", start.elapsed());
        assert_eq!(prover.verify(), Ok(()));

        // Reuse same k for apples-to-apples comparison on same evaluation domain
        let params: Params<EqAffine> = Params::new(k);

        let start = Instant::now();
        let vk = keygen_vk(&params, &p2_circuit).unwrap();
        let t_vk = start.elapsed();
        let start = Instant::now();
        let pk = keygen_pk(&params, vk, &p2_circuit).unwrap();
        let t_pk = start.elapsed();
        println!("  keygen_vk:    {:?}", t_vk);
        println!("  keygen_pk:    {:?}", t_pk);

        let start = Instant::now();
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        create_proof(
            &params,
            &pk,
            &[p2_circuit],
            &[&[]],
            OsRng,
            &mut transcript,
        )
        .unwrap();
        let p2_proof = transcript.finalize();
        println!("  create_proof: {:?}", start.elapsed());
        println!("  proof size:   {} bytes", p2_proof.len());

        let start = Instant::now();
        let strategy = SingleVerifier::new(&params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&p2_proof[..]);
        verify_proof(&params, pk.get_vk(), strategy, &[&[]], &mut transcript).unwrap();
        println!("  verify_proof: {:?}", start.elapsed());

        println!("\n========================================\n");
    }
}
