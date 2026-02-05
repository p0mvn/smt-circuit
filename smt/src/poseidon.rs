// Copyright (c) zkMove Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use ff::{FromUniformBytes, PrimeField};
use halo2_gadgets::poseidon::primitives::{generate_constants, ConstantLength, Hash, Mds, Spec};
use std::marker::PhantomData;

/// The same Poseidon specification as poseidon::P128Pow5T3
#[derive(Debug, Clone)]
pub struct SmtP128Pow5T3<F: PrimeField, const SECURE_MDS: usize>(PhantomData<F>);

impl<F: PrimeField, const SECURE_MDS: usize> SmtP128Pow5T3<F, SECURE_MDS> {
    pub fn new() -> Self {
        SmtP128Pow5T3(PhantomData::default())
    }
}

impl<F: PrimeField + FromUniformBytes<64> + Ord, const SECURE_MDS: usize> Spec<F, 3, 2>
    for SmtP128Pow5T3<F, SECURE_MDS>
{
    fn full_rounds() -> usize {
        8
    }

    fn partial_rounds() -> usize {
        56
    }

    fn sbox(val: F) -> F {
        val.pow_vartime([5])
    }

    fn secure_mds() -> usize {
        SECURE_MDS
    }

    fn constants() -> (Vec<[F; 3]>, Mds<F, 3>, Mds<F, 3>) {
        generate_constants::<_, Self, 3, 2>()
    }
}

impl<F: PrimeField, const SECURE_MDS: usize> Default for SmtP128Pow5T3<F, SECURE_MDS> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct Poseidon<F: PrimeField, const L: usize>(PhantomData<F>);

impl<F: PrimeField, const L: usize> Poseidon<F, L> {
    pub fn new() -> Self {
        Poseidon(PhantomData::default())
    }
}

pub trait FieldHasher<F: PrimeField, const L: usize> {
    fn hash(&self, inputs: [F; L]) -> Result<F>;
    fn hasher() -> Self;
}

impl<F, const L: usize> FieldHasher<F, L> for Poseidon<F, L>
where
    F: PrimeField + FromUniformBytes<64> + Ord,
{
    fn hash(&self, inputs: [F; L]) -> Result<F> {
        Ok(Hash::<_, SmtP128Pow5T3<F, 0>, ConstantLength<L>, 3, 2>::init().hash(inputs))
    }

    fn hasher() -> Self {
        Poseidon::<F, L>::default()
    }
}

impl<F: PrimeField, const L: usize> Default for Poseidon<F, L> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::poseidon::{FieldHasher, Poseidon, SmtP128Pow5T3};
    use halo2_gadgets::poseidon::primitives::Spec;
    use pasta_curves::Fp;

    #[test]
    fn poseidon_hash_basic() {
        let message = [Fp::from(6), Fp::from(42)];

        let poseidon = Poseidon::<Fp, 2>::new();
        let result = poseidon.hash(message).unwrap();

        // Hash should produce a non-zero result
        assert_ne!(result, Fp::from(0));

        // Same input should produce same output
        let result2 = poseidon.hash(message).unwrap();
        assert_eq!(result, result2);

        // Different input should produce different output
        let message2 = [Fp::from(7), Fp::from(42)];
        let result3 = poseidon.hash(message2).unwrap();
        assert_ne!(result, result3);
    }

    #[test]
    fn poseidon_spec_constants() {
        // Verify that constants can be generated
        let (round_constants, mds, mds_inv) = SmtP128Pow5T3::<Fp, 0>::constants();

        // Should have the correct number of round constants
        let full_rounds = SmtP128Pow5T3::<Fp, 0>::full_rounds();
        let partial_rounds = SmtP128Pow5T3::<Fp, 0>::partial_rounds();
        assert_eq!(round_constants.len(), full_rounds + partial_rounds);

        // MDS matrix should be 3x3
        assert_eq!(mds.len(), 3);
        assert_eq!(mds_inv.len(), 3);
    }
}
