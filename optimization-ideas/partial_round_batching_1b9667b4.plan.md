---
name: Partial Round Batching
overview: Batch 4 partial rounds per row in the Poseidon2 circuit chip (reducing rows per hash from 66 to 24), then add benchmarks comparing Poseidon1 vs Poseidon2 across native hashing and full proof generation.
todos:
  - id: batch-gate
    content: "Update poseidon2_chip.rs: expand rc to [Fixed;4], partial_sbox to [Advice;4], replace s_partial with s_partial_4 gate (symbolic 4-round chain), update hash() witness generation"
    status: pending
  - id: batch-tests
    content: Update poseidon2_chip unit tests for batched layout; run cargo test -p chiplet to verify all existing tests pass with the batched chip
    status: pending
  - id: bench-native
    content: Add native hash throughput benchmark (Poseidon1 vs Poseidon2, 10K hashes)
    status: pending
  - id: bench-circuit
    content: Add circuit proof benchmark (Poseidon1 via PoseidonChip vs Poseidon2-batched via Poseidon2Chip, HEIGHT=20, full proof cycle)
    status: pending
  - id: verify-all
    content: Run cargo test --workspace (release) and verify all tests + benchmarks pass end-to-end
    status: pending
isProject: false
---

# Partial Round Batching and Benchmarking

## Motivation

The current `Poseidon2Chip` uses 1 partial round per row, consuming 56 rows for the partial rounds section alone (66 total rows per hash). The reference implementation (`amit0365/poseidon2`) batches 4 partial rounds per row, reducing partial round rows from 56 to 14 (24 total rows per hash). Since the internal MDS is linear, all intermediate states can be expressed symbolically -- only the S-box outputs need witness columns.

## Row Count Comparison

```
                   Poseidon1 (Pow5Chip)   Poseidon2 (current)   Poseidon2 (batched)
Rows per hash      ~32                    66                    24
Partial round rows ~14 (2/row)            56 (1/row)            14 (4/row)
Max gate degree    6                      6                     6
```

The batching does not increase constraint degree (remains 6) because only one `x^5` S-box appears per batched round constraint, and the MDS operations between S-boxes are linear.

## 1. Batched Partial Round Gate Design

Replace the current `s_partial` gate (1 round per row) with `s_partial_4` (4 rounds per row).

### Column changes in `Poseidon2Config`

- `rc`: expand from `[Column<Fixed>; 3]` to `[Column<Fixed>; 4]`
  - Full rounds use `rc[0..3]` for the 3 state-element round constants (unchanged)
  - Batched partial rounds use `rc[0..4]` for the 4 per-round constants
- `partial_sbox`: expand from `Column<Advice>` to `[Column<Advice>; 4]`
  - 4 S-box witness outputs, one per batched round

### Gate constraint structure (7 polynomials, max degree 6)

The gate symbolically chains 4 partial rounds using `Expression<F>` arithmetic. Only the S-box outputs require witness columns; intermediate states are expressed as linear combinations:

```
// Build symbolic state through 4 rounds
let mut st = [cur[0], cur[1], cur[2]];
for i in 0..4 {
    // S-box constraint: psb[i] = (st[0] + rc[i])^5     [degree 6 with selector]
    st[0] = psb[i];
    // Internal MDS: sum = st[0]+st[1]+st[2]
    //   st = [st[0]+sum, st[1]+sum, st[2]*2+sum]        [linear, degree 1]
}
// State transition: next[i] = st[i]                      [degree 2 with selector]
```

4 S-box constraints (degree 6) + 3 state transition constraints (degree 2) = 7 total.

### Row layout (24 rows per hash)

```
Row 0:      s_first   (initial external linear layer)
Rows 1-4:   s_full    (first 4 full rounds, RC[0..4])
Rows 5-18:  s_partial_4  (14 batched rows x 4 = 56 partial rounds, RC[4..60])
Rows 19-22: s_full    (last 4 full rounds, RC[60..64])
Row 23:     (none)    final state, output = state[0]
```

### Witness generation changes in `hash()`

The witness loop changes from pushing 1 partial-round state per entry to pushing 1 state per 4-round batch. For each batch of 4 partial rounds, collect 4 S-box outputs and the resulting state after all 4 rounds.

## 2. Files Changed

- [chiplet/src/poseidon2_chip.rs](chiplet/src/poseidon2_chip.rs) -- Update `Poseidon2Config` columns, replace `s_partial` gate with `s_partial_4`, update witness generation in `hash()`, update unit tests
- [chiplet/src/smt_chip.rs](chiplet/src/smt_chip.rs) -- No production code changes needed (interface unchanged). Update `k` values in tests if any become too small (unlikely: all current k >= 10, and 24 rows/hash is even smaller than before)

## 3. Tests

- **Unit test: correctness** -- Update existing `test_poseidon2_chip_correct`: hash known inputs, verify MockProver output matches native `Poseidon2::hash()`. Reduce k from 10 to 7 (24 rows fits in 2^7=128).
- **Unit test: wrong output fails** -- Update existing `test_poseidon2_chip_wrong_output_fails`.
- **Integration: all existing smt_chip tests pass** -- `cargo test -p chiplet` (the Poseidon2Chip interface is unchanged, so PathChip/SparsePathChip work transparently).

## 4. Benchmarks

Add a new test `bench_poseidon1_vs_poseidon2` in [chiplet/src/smt_chip.rs](chiplet/src/smt_chip.rs) that measures:

### 4a. Native hash throughput

Hash 10,000 random input pairs with each hasher:

- `Poseidon<Fp, 2>` (Poseidon1 via halo2_gadgets primitives)
- `Poseidon2<Fp, 2>` (Poseidon2 native)

Print: total time, throughput (hashes/sec).

### 4b. Circuit proof (HEIGHT=20 Merkle path)

For each circuit variant, measure and print:

- MockProver time
- `keygen_vk` + `keygen_pk` time
- `create_proof` time
- `verify_proof` time
- Proof size (bytes)

Variants:

- **Poseidon1**: `TestCircuit` using original `PoseidonChip` + `SmtP128Pow5T3` + `Poseidon<Fp,2>` (import from preserved `poseidon_chip.rs`)
- **Poseidon2-batched**: `TestCircuit` using updated `Poseidon2Chip` + `Poseidon2<Fp,2>`

Both variants use k=13 (sufficient for HEIGHT=20 with either hash, provides apples-to-apples comparison on the same evaluation domain).

The benchmark reuses the existing test circuit structures (`TestConfig`/`TestCircuit`) for Poseidon2, and defines a parallel Poseidon1 circuit struct (similar to the old code, using `PoseidonChip<F, S, WIDTH, RATE, L>`) for comparison.