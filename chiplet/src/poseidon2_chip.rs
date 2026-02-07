// Copyright (c) zkMove Authors
// SPDX-License-Identifier: Apache-2.0

//! Poseidon2 circuit chip for the Pallas field.
//!
//! Self-contained implementation (no dependency on `halo2_gadgets::poseidon`).
//! Hardcoded for `t = 3`, `R_F = 8`, `R_P = 56`, `d = 5`.
//!
//! ## Gate layout (4 partial rounds batched per row)
//!
//! | Selector       | Rows           | Constraint                                           |
//! |----------------|----------------|------------------------------------------------------|
//! | `s_first`      | 0              | Initial external linear layer `circ(2,1,1)`          |
//! | `s_full`       | 1..=4, 19..=22 | Full round: add RC → S-box all → ext MDS             |
//! | `s_partial_4`  | 5..=18         | 4× partial round: add RC[0] → S-box[0] → int MDS    |
//! | (none)         | 23             | Final state (output = `state[0]`)                    |
//!
//! Total: **24 rows per hash** invocation.

use ff::PrimeField;
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error, Expression, Fixed, Selector},
    poly::Rotation,
};
use smt::poseidon2::{
    add_round_constants, matmul_external, matmul_internal, sbox, sbox_full, Poseidon2Params, R_F,
    R_P, ROUNDS,
};
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Config & Chip
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Poseidon2Config<F: PrimeField> {
    pub state: [Column<Advice>; 3],
    partial_sbox: [Column<Advice>; 4],
    rc: [Column<Fixed>; 4],
    s_first: Selector,
    s_full: Selector,
    s_partial_4: Selector,
    _marker: PhantomData<F>,
}

#[derive(Clone)]
pub struct Poseidon2Chip<F: PrimeField, const L: usize> {
    config: Poseidon2Config<F>,
    params: Poseidon2Params<F>,
}

impl<F: PrimeField, const L: usize> Poseidon2Chip<F, L> {
    pub fn construct(config: Poseidon2Config<F>) -> Self {
        Self {
            config,
            params: Poseidon2Params::new(),
        }
    }

    /// Configures the Poseidon2 chip by creating the columns and gates.
    pub fn configure(meta: &mut ConstraintSystem<F>) -> Poseidon2Config<F> {
        // Advice columns
        let state = [(); 3].map(|_| {
            let col = meta.advice_column();
            meta.enable_equality(col);
            col
        });
        let partial_sbox = [(); 4].map(|_| meta.advice_column());

        // Fixed columns (round constants + constant pool for domain tag)
        let rc = [(); 4].map(|_| meta.fixed_column());
        meta.enable_constant(rc[0]);

        // Selectors
        let s_first = meta.selector();
        let s_full = meta.selector();
        let s_partial_4 = meta.selector();

        // Helper: x^5 expression
        let pow5 = |x: Expression<F>| {
            let x2 = x.clone() * x.clone();
            let x4 = x2.clone() * x2;
            x4 * x
        };

        // ---- Gate: s_first  (initial external linear layer, circ(2,1,1)) ----
        //
        // sum       = cur[0] + cur[1] + cur[2]
        // next[i]   = cur[i] + sum
        // Constraint: next[i] - cur[i] - sum = 0
        meta.create_gate("poseidon2 initial linear layer", |meta| {
            let s = meta.query_selector(s_first);
            let cur: Vec<_> = (0..3)
                .map(|i| meta.query_advice(state[i], Rotation::cur()))
                .collect();
            let nxt: Vec<_> = (0..3)
                .map(|i| meta.query_advice(state[i], Rotation::next()))
                .collect();
            let sum = cur[0].clone() + cur[1].clone() + cur[2].clone();
            (0..3)
                .map(|i| s.clone() * (nxt[i].clone() - cur[i].clone() - sum.clone()))
                .collect::<Vec<_>>()
        });

        // ---- Gate: s_full  (full round) ----
        //
        // sb[i]     = (cur[i] + rc[i])^5
        // sum_sb    = sb[0] + sb[1] + sb[2]
        // next[i]   = sb[i] + sum_sb            (circ(2,1,1) on sboxed)
        // Constraint: next[i] - sb[i] - sum_sb = 0
        meta.create_gate("poseidon2 full round", |meta| {
            let s = meta.query_selector(s_full);
            let cur: Vec<_> = (0..3)
                .map(|i| meta.query_advice(state[i], Rotation::cur()))
                .collect();
            let nxt: Vec<_> = (0..3)
                .map(|i| meta.query_advice(state[i], Rotation::next()))
                .collect();
            let rc_e: Vec<_> = (0..3)
                .map(|i| meta.query_fixed(rc[i]))
                .collect();

            let sb: Vec<_> = (0..3)
                .map(|i| pow5(cur[i].clone() + rc_e[i].clone()))
                .collect();
            let sum_sb = sb[0].clone() + sb[1].clone() + sb[2].clone();

            (0..3)
                .map(|i| s.clone() * (nxt[i].clone() - sb[i].clone() - sum_sb.clone()))
                .collect::<Vec<_>>()
        });

        // ---- Gate: s_partial_4  (4 batched partial rounds per row) ----
        //
        // Symbolically chains 4 partial rounds using Expression<F> arithmetic.
        // Only the S-box outputs (partial_sbox[0..4]) require witness columns;
        // intermediate states are expressed as linear combinations.
        //
        // For each of the 4 rounds:
        //   S-box constraint: psb[i] = (st[0] + rc[i])^5   [degree 6 with selector]
        //   st[0] <- psb[i]
        //   Internal MDS: sum = st[0]+st[1]+st[2]
        //     st[0] <- st[0]+sum; st[1] <- st[1]+sum; st[2] <- 2*st[2]+sum
        //
        // Then: nxt[j] = st[j]  (state transition)          [degree 2 with selector]
        //
        // Total: 4 S-box checks (degree 6) + 3 transitions (degree 2) = 7 constraints.
        meta.create_gate("poseidon2 partial round x4", |meta| {
            let s = meta.query_selector(s_partial_4);
            let cur: Vec<_> = (0..3)
                .map(|i| meta.query_advice(state[i], Rotation::cur()))
                .collect();
            let nxt: Vec<_> = (0..3)
                .map(|i| meta.query_advice(state[i], Rotation::next()))
                .collect();
            let rc_vals: Vec<_> = (0..4)
                .map(|i| meta.query_fixed(rc[i]))
                .collect();
            let psb: Vec<_> = (0..4)
                .map(|i| meta.query_advice(partial_sbox[i], Rotation::cur()))
                .collect();

            let two = Expression::Constant(F::from(2));

            // Build symbolic state through 4 partial rounds
            let mut st = [cur[0].clone(), cur[1].clone(), cur[2].clone()];
            let mut constraints = Vec::with_capacity(7);

            for i in 0..4 {
                // S-box constraint: psb[i] = (st[0] + rc[i])^5
                constraints.push(
                    s.clone() * (psb[i].clone() - pow5(st[0].clone() + rc_vals[i].clone())),
                );

                // Replace state[0] with S-box witness output
                st[0] = psb[i].clone();

                // Internal MDS: M = I + diag(1,1,2)  ->  [[2,1,1],[1,2,1],[1,1,3]]
                let sum = st[0].clone() + st[1].clone() + st[2].clone();
                st = [
                    st[0].clone() + sum.clone(),
                    st[1].clone() + sum.clone(),
                    st[2].clone() * two.clone() + sum,
                ];
            }

            // State transition constraints: nxt[j] = st[j]
            for i in 0..3 {
                constraints.push(s.clone() * (nxt[i].clone() - st[i].clone()));
            }

            constraints
        });

        Poseidon2Config {
            state,
            partial_sbox,
            rc,
            s_first,
            s_full,
            s_partial_4,
            _marker: PhantomData,
        }
    }

    /// Compute a Poseidon2 hash in-circuit.
    ///
    /// Accepts `L` input cells and returns the hash output cell.
    /// Internally runs the sponge construction: domain-separated initial state,
    /// one full permutation, squeeze `state[0]`.
    pub fn hash(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: &[AssignedCell<F, F>; L],
    ) -> Result<AssignedCell<F, F>, Error> {
        let params = &self.params;

        layouter.assign_region(
            || "poseidon2 hash",
            |mut region| {
                // ----- Precompute witness values --------------------------------
                // `all_states[i]` = state at row i  (24 entries, indices 0..=23)
                // `p_sboxes[j]`   = S-box output at partial round j (56 entries)
                let witness = inputs[0]
                    .value()
                    .copied()
                    .zip(inputs[1].value().copied())
                    .map(|(a, b)| {
                        let mut st = [a, b, F::from(L as u64)];
                        let mut all: Vec<[F; 3]> = Vec::with_capacity(24);
                        let mut psb: Vec<F> = Vec::with_capacity(R_P);

                        all.push(st); // row 0

                        // Initial external linear layer -> row 1
                        matmul_external(&mut st);
                        all.push(st);

                        let rh = R_F / 2; // 4

                        // First 4 full rounds -> rows 2..=5
                        for r in 0..rh {
                            add_round_constants(&mut st, &params.round_constants[r]);
                            sbox_full(&mut st);
                            matmul_external(&mut st);
                            all.push(st);
                        }

                        // 14 batches of 4 partial rounds -> rows 6..=19
                        for batch in 0..14usize {
                            for i in 0..4usize {
                                let r = rh + batch * 4 + i;
                                st[0] += params.round_constants[r][0];
                                let sb = sbox(st[0]);
                                psb.push(sb);
                                st[0] = sb;
                                matmul_internal(&mut st, &params.mat_internal_diag_m_1);
                            }
                            all.push(st);
                        }

                        // Last 4 full rounds -> rows 20..=23
                        for r in (rh + R_P)..ROUNDS {
                            add_round_constants(&mut st, &params.round_constants[r]);
                            sbox_full(&mut st);
                            matmul_external(&mut st);
                            all.push(st);
                        }

                        (all, psb)
                    });

                // Convenience helpers
                let state_val = |row: usize, col: usize| -> Value<F> {
                    witness.as_ref().map(|(s, _)| s[row][col])
                };
                let sbox_val = |idx: usize| -> Value<F> {
                    witness.as_ref().map(|(_, p)| p[idx])
                };

                // ----- Row 0: initial state + s_first ----------------------------
                inputs[0].copy_advice(
                    || "input[0]",
                    &mut region,
                    self.config.state[0],
                    0,
                )?;
                inputs[1].copy_advice(
                    || "input[1]",
                    &mut region,
                    self.config.state[1],
                    0,
                )?;
                region.assign_advice_from_constant(
                    || "domain tag",
                    self.config.state[2],
                    0,
                    F::from(L as u64),
                )?;
                self.config.s_first.enable(&mut region, 0)?;

                // ----- Rows 1..=23: states + selectors + round constants ----------
                let mut output_cell: Option<AssignedCell<F, F>> = None;

                for row in 1..=23usize {
                    // Assign state[0..3]
                    let cell0 = region.assign_advice(
                        || format!("s0 r{}", row),
                        self.config.state[0],
                        row,
                        || state_val(row, 0),
                    )?;
                    region.assign_advice(
                        || format!("s1 r{}", row),
                        self.config.state[1],
                        row,
                        || state_val(row, 1),
                    )?;
                    region.assign_advice(
                        || format!("s2 r{}", row),
                        self.config.state[2],
                        row,
                        || state_val(row, 2),
                    )?;

                    // The last row (23) is final output -- no selector needed
                    if row <= 22 {
                        if row <= 4 || row >= 19 {
                            // ---- Full round ----
                            self.config.s_full.enable(&mut region, row)?;
                            let round_idx = if row <= 4 {
                                row - 1
                            } else {
                                60 + (row - 19)
                            };
                            for j in 0..3 {
                                region.assign_fixed(
                                    || format!("rc{} r{}", j, row),
                                    self.config.rc[j],
                                    row,
                                    || Value::known(params.round_constants[round_idx][j]),
                                )?;
                            }
                        } else {
                            // ---- Batched partial round (4 rounds per row) ----
                            self.config.s_partial_4.enable(&mut region, row)?;
                            let batch_idx = row - 5;
                            let base_round = 4 + batch_idx * 4;
                            // Assign 4 round constants (one per batched round)
                            for j in 0..4 {
                                region.assign_fixed(
                                    || format!("rc{} r{}", j, row),
                                    self.config.rc[j],
                                    row,
                                    || {
                                        Value::known(
                                            params.round_constants[base_round + j][0],
                                        )
                                    },
                                )?;
                            }
                            // Assign 4 S-box witness outputs
                            let sbox_base = batch_idx * 4;
                            for j in 0..4 {
                                region.assign_advice(
                                    || format!("psb{} r{}", j, row),
                                    self.config.partial_sbox[j],
                                    row,
                                    || sbox_val(sbox_base + j),
                                )?;
                            }
                        }
                    }

                    if row == 23 {
                        output_cell = Some(cell0);
                    }
                }

                Ok(output_cell.unwrap())
            },
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{Poseidon2Chip, Poseidon2Config};
    use ff::PrimeField;
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::{
        circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
        plonk::{Advice, Circuit, Column, ConstraintSystem, Error},
    };
    use pasta_curves::Fp;
    use smt::poseidon::FieldHasher;
    use smt::poseidon2::Poseidon2;

    // ---- Test circuit: hash two inputs and constrain against expected output --

    #[derive(Clone)]
    struct TestConfig<F: PrimeField> {
        poseidon2_config: Poseidon2Config<F>,
        inputs: [Column<Advice>; 2],
        output: Column<Advice>,
    }

    struct TestCircuit<F: PrimeField> {
        a: Value<F>,
        b: Value<F>,
        expected: Value<F>,
    }

    impl<F: PrimeField> Circuit<F> for TestCircuit<F> {
        type Config = TestConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                a: Value::default(),
                b: Value::default(),
                expected: Value::default(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let inputs = [meta.advice_column(), meta.advice_column()];
            let output = meta.advice_column();
            inputs
                .iter()
                .for_each(|c| meta.enable_equality(*c));
            meta.enable_equality(output);

            TestConfig {
                poseidon2_config: Poseidon2Chip::<F, 2>::configure(meta),
                inputs,
                output,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            // Assign inputs
            let assigned_inputs = layouter.assign_region(
                || "assign inputs",
                |mut region| -> Result<[AssignedCell<F, F>; 2], Error> {
                    let a = region.assign_advice(
                        || "a",
                        config.inputs[0],
                        0,
                        || self.a,
                    )?;
                    let b = region.assign_advice(
                        || "b",
                        config.inputs[1],
                        0,
                        || self.b,
                    )?;
                    Ok([a, b])
                },
            )?;

            // Hash
            let chip = Poseidon2Chip::<F, 2>::construct(config.poseidon2_config.clone());
            let hash_cell =
                chip.hash(&mut layouter.namespace(|| "hash"), &assigned_inputs)?;

            // Constrain output
            layouter.assign_region(
                || "constrain output",
                |mut region| {
                    let expected_cell = region.assign_advice(
                        || "expected",
                        config.output,
                        0,
                        || self.expected,
                    )?;
                    region.constrain_equal(hash_cell.cell(), expected_cell.cell())
                },
            )
        }
    }

    #[test]
    fn test_poseidon2_chip_correct() {
        let k = 7; // 24 rows per hash, 2^7 = 128 rows available

        let a = Fp::from(3);
        let b = Fp::from(2);
        let hasher = Poseidon2::<Fp, 2>::new();
        let c = hasher.hash([a, b]).unwrap();

        let circuit = TestCircuit::<Fp> {
            a: Value::known(a),
            b: Value::known(b),
            expected: Value::known(c),
        };

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn test_poseidon2_chip_wrong_output_fails() {
        let k = 7;

        let a = Fp::from(3);
        let b = Fp::from(2);
        let wrong_c = Fp::from(42);

        let circuit = TestCircuit::<Fp> {
            a: Value::known(a),
            b: Value::known(b),
            expected: Value::known(wrong_c),
        };

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}
