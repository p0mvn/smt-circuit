---
name: Poseidon2 native replacement
overview: Replace the native Poseidon1 hash implementation in the `smt` crate with Poseidon2, while preserving the `FieldHasher` trait interface and keeping the circuit (chiplet) Poseidon1 chip unchanged. All existing tests must pass with the new hash function.
todos:
  - id: generate-params
    content: Generate Pallas Poseidon2 parameters using Sage script; create smt/src/poseidon2_params.rs with round constants and internal matrix diagonal
    status: completed
  - id: port-permutation
    content: Port Poseidon2 permutation algorithm from ark_ff to ff::PrimeField in smt/src/poseidon2.rs
    status: completed
  - id: implement-sponge
    content: Implement sponge construction (absorb/squeeze) for ConstantLength hashing over the Poseidon2 permutation
    status: completed
  - id: implement-field-hasher
    content: Implement FieldHasher<F, L> for Poseidon2<F, L> struct; update smt/src/lib.rs exports
    status: completed
  - id: add-poseidon2-tests
    content: Add KAT test (Pallas [0,1,2] vector), determinism test, collision test, and sponge correctness test
    status: completed
  - id: update-smt-tests
    content: Switch smt/src/smt.rs and smt/src/compressed_smt.rs test modules from Poseidon<Fp,2> to Poseidon2<Fp,2>
    status: in_progress
  - id: verify-full-suite
    content: Run cargo test --workspace and verify zero failures
    status: pending
isProject: false
---

# Poseidon2 Native Drop-in Replacement

## Context

The current codebase uses Poseidon1 via `halo2_gadgets::poseidon::primitives` in `[smt/src/poseidon.rs](smt/src/poseidon.rs)`. The `FieldHasher` trait is consumed by:

- `[smt/src/smt.rs](smt/src/smt.rs)` -- `SparseMerkleTree`, `Path`, `SparsePath`, `gen_empty_hashes`
- `[smt/src/compressed_smt.rs](smt/src/compressed_smt.rs)` -- `CompressedSMT`, `build`, `fill_path`, etc.
- `[chiplet/src/smt_chip.rs](chiplet/src/smt_chip.rs)` -- test helpers only (native hash for generating witness data)

The `FieldHasher` trait interface stays unchanged:

```rust
pub trait FieldHasher<F: PrimeField, const L: usize> {
    fn hash(&self, inputs: [F; L]) -> Result<F>;
    fn hasher() -> Self;
}
```

The circuit chip in `[chiplet/src/poseidon_chip.rs](chiplet/src/poseidon_chip.rs)` remains on Poseidon1 for now (it depends on `halo2_gadgets::poseidon::Pow5Chip`). The `SmtP128Pow5T3` spec also stays since it is used by the chiplet crate for circuit proofs.

## Key Compatibility Notes

- The Poseidon2 reference repo ([amit0365/poseidon2](https://github.com/amit0365/poseidon2)) uses `ark_ff::PrimeField`. Our codebase uses `ff::PrimeField`. The algorithm must be ported to the `ff` trait ecosystem.
- The Sage script already has the Pallas prime defined: `p = 28948022309329048855892746252171976963363056481941560715954676764349967630337`. The reference repo has commented-out Pallas KAT values we can use to verify correctness.
- For t=3 (WIDTH=3), the Poseidon2 parameters are: `R_F=8, R_P=56, d=5` (same count as Poseidon1 but different algorithm structure).
- Since hash outputs will differ from Poseidon1, all root values change. Existing tests are structural (random leaves, internal consistency checks) so they will pass with a different hash function.

## Implementation Steps

### 1. Generate Pallas Poseidon2 Parameters via Sage

Run the `poseidon2_rust_params.sage` script (from the reference repo) with the Pallas prime uncommented and `t=3`. This produces:

- Round constants: `(R_F * t) + R_P = (8 * 3) + 56 = 80` field elements, organized as `Vec<[F; 3]>` (full rounds get 3 constants, partial rounds get 1 constant + 2 zeros)
- Internal matrix diagonal minus identity: `mat_internal_diag_m_1` (3 field elements)
- External matrix: hardcoded `circ(2, 1, 1)` for t=3

Store these in a new file `[smt/src/poseidon2_params.rs](smt/src/poseidon2_params.rs)` as `lazy_static` or `const` arrays of hex-encoded `Fp` values.

### 2. Port the Poseidon2 Permutation Algorithm

Create `[smt/src/poseidon2.rs](smt/src/poseidon2.rs)` containing:

- `**Poseidon2Params<F>` struct** -- holds `t`, `d`, round counts, round constants, `mat_internal_diag_m_1`
- `**Poseidon2Permutation<F>` struct** -- wraps params, implements the permutation:
  1. Initial external linear layer: `matmul_external()`
  2. First half full rounds: add RC, S-box (`x^5`) on all elements, `matmul_external()`
  3. Partial rounds: add RC to first element only, S-box on first element, `matmul_internal()`
  4. Second half full rounds: add RC, S-box on all elements, `matmul_external()`

For t=3, `matmul_external` is `circ(2,1,1)`: each output = `2*self + sum(all)`. `matmul_internal` for t=3 is the hardcoded `[[2,1,1],[1,2,1],[1,1,3]]` matrix using the diagonal-minus-1 form.

Key adaptation from `ark_ff` to `ff`:

- `square_in_place()` becomes `let sq = val.square()`
- `mul_assign()` becomes `val *= &other` or `val * other`  
- `add_assign()` becomes `val += &other`
- `F::zero()` becomes `F::ZERO`

### 3. Implement the Sponge Construction

Implement a sponge hash on top of the permutation in the same file. For `ConstantLength<L>` with WIDTH=3, RATE=2:

```rust
fn hash_sponge(inputs: [F; L]) -> F {
    // Initialize state: [0, 0, 0]
    let mut state = [F::ZERO; 3];
    // Domain separation: capacity = encoded length
    state[2] = F::from(L as u64);  // or appropriate domain tag
    // Absorb: add inputs to rate portion
    for (i, chunk) in inputs.chunks(2).enumerate() {
        for (j, &input) in chunk.iter().enumerate() {
            state[j] += input;
        }
        state = permutation(state);
    }
    // Squeeze: return first element
    state[0]
}
```

Note: The domain separation strategy should be documented. We use the same approach as halo2's `ConstantLength`: the capacity element encodes the input length.

### 4. Create `Poseidon2<F, L>` Implementing `FieldHasher`

In `[smt/src/poseidon2.rs](smt/src/poseidon2.rs)`:

```rust
pub struct Poseidon2<F: PrimeField, const L: usize>(PhantomData<F>);

impl<F, const L: usize> FieldHasher<F, L> for Poseidon2<F, L>
where F: PrimeField + FromUniformBytes<64> + Ord
{
    fn hash(&self, inputs: [F; L]) -> Result<F> {
        // Use Poseidon2 sponge
        Ok(poseidon2_hash::<F, L>(inputs))
    }
    fn hasher() -> Self { Poseidon2::default() }
}
```

### 5. Update Module Structure

In `[smt/src/lib.rs](smt/src/lib.rs)`, add:

```rust
pub mod poseidon2;
pub mod poseidon2_params;
```

Keep `pub mod poseidon;` -- the existing Poseidon1 module stays since the chiplet crate imports `SmtP128Pow5T3` for circuit proofs.

### 6. Update Imports in Consumers (Optional for Phase 1)

For the initial phase, tests and consumers can be updated to use `Poseidon2<Fp, 2>` instead of `Poseidon<Fp, 2>`. This is a search-and-replace in test modules of:

- `smt/src/smt.rs` -- test module
- `smt/src/compressed_smt.rs` -- test module
- `chiplet/src/smt_chip.rs` -- test module (for witness generation only; the circuit still uses Poseidon1 Pow5Chip)

Alternatively, re-export `Poseidon2` as `Poseidon` from `poseidon.rs` for minimal diff, but this is less clear.

### 7. Add Dedicated Poseidon2 Tests

In `[smt/src/poseidon2.rs](smt/src/poseidon2.rs)` tests:

- **KAT test**: Verify permutation output for input `[0, 1, 2]` matches the commented-out Pallas values from the reference repo:
  ```
  perm[0] == 0x1a9b54c7512a914dd778282c44b3513fea7251420b9d95750baae059b2268d7a
  perm[1] == 0x1c48ea0994a7d7984ea338a54dbf0c8681f5af883fe988d59ba3380c9f7901fc
  perm[2] == 0x079ddd0a80a3e9414489b526a2770448964766685f4c4842c838f8a23120b401
  ```
- **Determinism test**: Same input produces same output
- **Collision test**: Different inputs produce different outputs
- **Sponge test**: Verify `FieldHasher::hash([a, b])` returns deterministic, non-zero output

### 8. Dependency Changes

In `[smt/Cargo.toml](smt/Cargo.toml)`, no new dependencies are needed. The algorithm is implemented purely using `ff::PrimeField` arithmetic. Remove the `halo2_gadgets` dependency from `smt` if it's only used for the Poseidon1 hash (check if it's also used elsewhere). Actually, `halo2_gadgets` must stay because `SmtP128Pow5T3` implements `halo2_gadgets::poseidon::primitives::Spec` which the chiplet still needs.

### 9. Run Full Test Suite

```bash
cargo test -p smt          # All SMT + compressed SMT tests
cargo test -p chiplet      # All circuit tests (still uses Poseidon1 for proofs)
```

The chiplet tests use both `Poseidon<Fp, 2>` (for witness generation) and `SmtP128Pow5T3` (for circuit proofs). For phase 1 where the circuit stays on Poseidon1, the chiplet tests must continue using `Poseidon<Fp, 2>` (Poseidon1) for witness generation to match circuit constraints. Only `smt` crate tests switch to Poseidon2.

## File Summary


| File                           | Action                                                         |
| ------------------------------ | -------------------------------------------------------------- |
| `smt/src/poseidon2_params.rs`  | **New** -- Pallas Poseidon2 round constants and matrix         |
| `smt/src/poseidon2.rs`         | **New** -- Poseidon2 permutation, sponge, `FieldHasher` impl   |
| `smt/src/lib.rs`               | **Edit** -- add `pub mod poseidon2; pub mod poseidon2_params;` |
| `smt/src/poseidon.rs`          | **Keep** -- unchanged, still needed by chiplet                 |
| `smt/src/smt.rs`               | **Edit** -- test module: switch to `Poseidon2<Fp, 2>`          |
| `smt/src/compressed_smt.rs`    | **Edit** -- test module: switch to `Poseidon2<Fp, 2>`          |
| `chiplet/src/smt_chip.rs`      | **No change** -- keeps Poseidon1 for both circuit and witness  |
| `chiplet/src/poseidon_chip.rs` | **No change**                                                  |


## Verification Checklist

- Poseidon2 KAT matches reference implementation (Pallas `[0,1,2]` test vector)
- All `smt` crate tests pass with `Poseidon2`
- All `chiplet` crate tests pass unchanged (still on Poseidon1)
- `cargo test --workspace` passes with zero failures

