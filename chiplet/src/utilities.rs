// Copyright (c) zkMove Authors
// SPDX-License-Identifier: Apache-2.0

use ff::PrimeField;
use halo2_proofs::{
    circuit::{AssignedCell, Chip, Layouter, Region},
    plonk::{Advice, Column, ConstraintSystem, Error, Expression, Selector},
    poly::Rotation,
};
use std::marker::PhantomData;

pub const NUM_OF_UTILITY_ADVICE_COLUMNS: usize = 4;
pub const NUM_OF_SWAP_ADVICE_COLUMNS: usize = 5;

#[derive(Clone, Debug)]
pub struct ConditionalSelectConfig<F: PrimeField> {
    advices: [Column<Advice>; NUM_OF_UTILITY_ADVICE_COLUMNS],
    s_cs: Selector,
    _marker: PhantomData<F>,
}

pub struct ConditionalSelectChip<F: PrimeField> {
    config: ConditionalSelectConfig<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Chip<F> for ConditionalSelectChip<F> {
    type Config = ConditionalSelectConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: PrimeField> ConditionalSelectChip<F> {
    pub fn construct(
        config: <Self as Chip<F>>::Config,
        _loaded: <Self as Chip<F>>::Loaded,
    ) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        advices: [Column<Advice>; NUM_OF_UTILITY_ADVICE_COLUMNS],
    ) -> <Self as Chip<F>>::Config {
        for column in &advices {
            meta.enable_equality(*column);
        }
        let s_cs = meta.selector();

        meta.create_gate("conditional_select", |meta| {
            let lhs = meta.query_advice(advices[0], Rotation::cur());
            let rhs = meta.query_advice(advices[1], Rotation::cur());
            let out = meta.query_advice(advices[2], Rotation::cur());
            let cond = meta.query_advice(advices[3], Rotation::cur());
            let s_cs = meta.query_selector(s_cs);
            let one = Expression::Constant(F::ONE);

            vec![
                // cond is 0 or 1
                s_cs.clone() * (cond.clone() * (one - cond.clone())),
                // lhs * cond + rhs * (1 - cond) = out
                s_cs * ((lhs - rhs.clone()) * cond + rhs - out),
            ]
        });

        ConditionalSelectConfig {
            advices,
            s_cs,
            _marker: PhantomData,
        }
    }

    pub fn conditional_select(
        &self,
        layouter: &mut impl Layouter<F>,
        a: AssignedCell<F, F>,
        b: AssignedCell<F, F>,
        cond: AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, Error> {
        let config = self.config();
        let out = layouter.assign_region(
            || "conditional_select",
            |mut region: Region<'_, F>| {
                config.s_cs.enable(&mut region, 0)?;

                a.copy_advice(|| "copy a", &mut region, config.advices[0], 0)?;

                b.copy_advice(|| "copy b", &mut region, config.advices[1], 0)?;

                let cond = cond.copy_advice(|| "copy cond", &mut region, config.advices[3], 0)?;

                // Use zip_map to combine values and select based on condition
                let selected = cond.value().copied().and_then(|c| {
                    if c == F::ONE {
                        a.value().copied()
                    } else {
                        b.value().copied()
                    }
                });

                let cell =
                    region.assign_advice(|| "select result", config.advices[2], 0, || selected)?;
                Ok(cell)
            },
        )?;
        Ok(out)
    }
}

#[derive(Clone, Debug)]
pub struct IsEqualConfig<F: PrimeField> {
    s_is_eq: Selector,
    advices: [Column<Advice>; NUM_OF_UTILITY_ADVICE_COLUMNS],
    _marker: PhantomData<F>,
}

pub struct IsEqualChip<F: PrimeField> {
    config: IsEqualConfig<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Chip<F> for IsEqualChip<F> {
    type Config = IsEqualConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: PrimeField> IsEqualChip<F> {
    pub fn construct(
        config: <Self as Chip<F>>::Config,
        _loaded: <Self as Chip<F>>::Loaded,
    ) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        advices: [Column<Advice>; NUM_OF_UTILITY_ADVICE_COLUMNS],
    ) -> <Self as Chip<F>>::Config {
        let s_is_eq = meta.selector();
        meta.create_gate("eq", |meta| {
            let lhs = meta.query_advice(advices[0], Rotation::cur());
            let rhs = meta.query_advice(advices[1], Rotation::cur());
            let out = meta.query_advice(advices[2], Rotation::cur());
            let delta_invert = meta.query_advice(advices[3], Rotation::cur());
            let s_is_eq = meta.query_selector(s_is_eq);
            let one = Expression::Constant(F::ONE);

            vec![
                // out is 0 or 1
                s_is_eq.clone() * (out.clone() * (one.clone() - out.clone())),
                // if a != b then (a - b) * inverse(a - b) == 1 - out
                // if a == b then (a - b) * 1 == 1 - out
                s_is_eq.clone()
                    * ((lhs.clone() - rhs.clone()) * delta_invert.clone() + (out - one.clone())),
                // constrain delta_invert: (a - b) * inverse(a - b) must be 1 or 0
                s_is_eq * (lhs.clone() - rhs.clone()) * ((lhs - rhs) * delta_invert - one),
            ]
        });

        IsEqualConfig {
            s_is_eq,
            advices,
            _marker: PhantomData,
        }
    }

    pub fn is_eq_with_output(
        &self,
        layouter: &mut impl Layouter<F>,
        a: AssignedCell<F, F>,
        b: AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, Error> {
        let config = self.config();

        let out = layouter.assign_region(
            || "is_eq",
            |mut region: Region<'_, F>| {
                config.s_is_eq.enable(&mut region, 0)?;

                a.copy_advice(|| "copy a", &mut region, config.advices[0], 0)?;
                b.copy_advice(|| "copy b", &mut region, config.advices[1], 0)?;

                // Compute delta_invert and is_eq using Value combinators
                let delta_invert_val = a.value().copied().zip(b.value().copied()).map(|(a_val, b_val)| {
                    let delta = a_val - b_val;
                    if delta == F::ZERO {
                        F::ONE
                    } else {
                        delta.invert().unwrap_or(F::ONE)
                    }
                });

                region.assign_advice(
                    || "delta invert",
                    config.advices[3],
                    0,
                    || delta_invert_val,
                )?;

                let is_eq = a.value().copied().zip(b.value().copied()).map(|(a_val, b_val)| {
                    if a_val == b_val {
                        F::ONE
                    } else {
                        F::ZERO
                    }
                });

                let cell = region.assign_advice(|| "is_eq", config.advices[2], 0, || is_eq)?;
                Ok(cell)
            },
        )?;

        Ok(out)
    }
}

#[derive(Clone, Debug)]
pub struct AssertEqualConfig<F: PrimeField> {
    s_eq: Selector,
    advices: [Column<Advice>; 2],
    _marker: PhantomData<F>,
}

pub struct AssertEqualChip<F: PrimeField> {
    config: AssertEqualConfig<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Chip<F> for AssertEqualChip<F> {
    type Config = AssertEqualConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: PrimeField> AssertEqualChip<F> {
    pub fn construct(
        config: <Self as Chip<F>>::Config,
        _loaded: <Self as Chip<F>>::Loaded,
    ) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        advices: [Column<Advice>; 2],
    ) -> <Self as Chip<F>>::Config {
        let s_eq = meta.selector();
        meta.create_gate("eq", |meta| {
            let lhs = meta.query_advice(advices[0], Rotation::cur());
            let rhs = meta.query_advice(advices[1], Rotation::cur());
            let s_eq = meta.query_selector(s_eq);

            vec![s_eq * (lhs - rhs)]
        });

        AssertEqualConfig {
            s_eq,
            advices,
            _marker: PhantomData,
        }
    }

    pub fn assert_equal(
        &self,
        layouter: &mut impl Layouter<F>,
        a: AssignedCell<F, F>,
        b: AssignedCell<F, F>,
    ) -> Result<(), Error> {
        let config = self.config();

        layouter.assign_region(
            || "is_eq",
            |mut region: Region<'_, F>| {
                config.s_eq.enable(&mut region, 0)?;

                a.copy_advice(|| "copy a", &mut region, config.advices[0], 0)?;
                b.copy_advice(|| "copy b", &mut region, config.advices[1], 0)?;
                Ok(())
            },
        )?;

        Ok(())
    }
}

/// A single-row conditional swap gate.
///
/// Given inputs `(a, b, bit)`, outputs `(out_a, out_b)` where:
/// - `bit = 0`: `(out_a, out_b) = (a, b)` (no swap)
/// - `bit = 1`: `(out_a, out_b) = (b, a)` (swapped)
///
/// Uses 5 advice columns in a single row with constraints:
/// - `bit * (1 - bit) = 0`              (boolean)
/// - `out_a = (1 - bit) * a + bit * b`  (first output)
/// - `out_b = bit * a + (1 - bit) * b`  (second output)
#[derive(Clone, Debug)]
pub struct ConditionalSwapConfig<F: PrimeField> {
    advices: [Column<Advice>; NUM_OF_SWAP_ADVICE_COLUMNS],
    s_swap: Selector,
    _marker: PhantomData<F>,
}

pub struct ConditionalSwapChip<F: PrimeField> {
    config: ConditionalSwapConfig<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Chip<F> for ConditionalSwapChip<F> {
    type Config = ConditionalSwapConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: PrimeField> ConditionalSwapChip<F> {
    pub fn construct(
        config: <Self as Chip<F>>::Config,
        _loaded: <Self as Chip<F>>::Loaded,
    ) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        advices: [Column<Advice>; NUM_OF_SWAP_ADVICE_COLUMNS],
    ) -> <Self as Chip<F>>::Config {
        for column in &advices {
            meta.enable_equality(*column);
        }
        let s_swap = meta.selector();

        meta.create_gate("conditional_swap", |meta| {
            let a = meta.query_advice(advices[0], Rotation::cur());
            let b = meta.query_advice(advices[1], Rotation::cur());
            let bit = meta.query_advice(advices[2], Rotation::cur());
            let out_a = meta.query_advice(advices[3], Rotation::cur());
            let out_b = meta.query_advice(advices[4], Rotation::cur());
            let s_swap = meta.query_selector(s_swap);
            let one = Expression::Constant(F::ONE);

            vec![
                // bit is boolean
                s_swap.clone() * (bit.clone() * (one.clone() - bit.clone())),
                // out_a = (1 - bit) * a + bit * b
                s_swap.clone()
                    * (out_a
                        - (one.clone() - bit.clone()) * a.clone()
                        - bit.clone() * b.clone()),
                // out_b = bit * a + (1 - bit) * b
                s_swap * (out_b - bit.clone() * a - (one - bit) * b),
            ]
        });

        ConditionalSwapConfig {
            advices,
            s_swap,
            _marker: PhantomData,
        }
    }

    pub fn swap(
        &self,
        layouter: &mut impl Layouter<F>,
        a: AssignedCell<F, F>,
        b: AssignedCell<F, F>,
        bit: AssignedCell<F, F>,
    ) -> Result<(AssignedCell<F, F>, AssignedCell<F, F>), Error> {
        let config = self.config();

        layouter.assign_region(
            || "conditional_swap",
            |mut region: Region<'_, F>| {
                config.s_swap.enable(&mut region, 0)?;

                a.copy_advice(|| "copy a", &mut region, config.advices[0], 0)?;
                b.copy_advice(|| "copy b", &mut region, config.advices[1], 0)?;
                bit.copy_advice(|| "copy bit", &mut region, config.advices[2], 0)?;

                let out_a_val = bit.value().copied().and_then(|bit_val| {
                    if bit_val == F::ZERO {
                        a.value().copied()
                    } else {
                        b.value().copied()
                    }
                });

                let out_b_val = bit.value().copied().and_then(|bit_val| {
                    if bit_val == F::ZERO {
                        b.value().copied()
                    } else {
                        a.value().copied()
                    }
                });

                let out_a =
                    region.assign_advice(|| "out_a", config.advices[3], 0, || out_a_val)?;
                let out_b =
                    region.assign_advice(|| "out_b", config.advices[4], 0, || out_b_val)?;

                Ok((out_a, out_b))
            },
        )
    }
}
