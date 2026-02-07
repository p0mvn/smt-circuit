# Custom Optimizations

## Direction Bit SMT Optimization

**Goal:** Reduce per-level overhead in the SMT circuit's `calculate_root` from 3 rows to 1 row.

### Problem

Each level of the Merkle path verification used three utility gadgets (3 rows per level):

1. `IsEqual(prev, left)` -- determine which side `prev` is on
2. `ConditionalSelect(left, right, cond)` -- select the matching side
3. `AssertEqual(prev, result)` -- verify the selection

These are redundant because the prover already knows the direction at each level during proof generation.

### Solution

Replaced the three-gadget pattern with a single **ConditionalSwap** gate. The prover supplies a direction bit per level indicating whether the path descends left (0) or right (1). The swap gate reorders `(prev_hash, sibling)` into the correct `(left, right)` hash inputs in a single row:

- `bit = 0` (left child): `hash(prev, sibling)` -- no swap
- `bit = 1` (right child): `hash(sibling, prev)` -- swapped

The gate uses 5 advice columns with 3 constraints:

```
bit * (1 - bit) = 0                          (boolean check)
out_a = (1 - bit) * a + bit * b              (first output)
out_b = bit * a + (1 - bit) * b              (second output)
```

### Files Changed

| File | Change |
|------|--------|
| `smt/src/smt.rs` | Added `direction_bits: [bool; N]` to `Path`, populated in `generate_membership_proof` |
| `chiplet/src/utilities.rs` | Added `ConditionalSwapChip` with single-row 5-column gate |
| `chiplet/src/smt_chip.rs` | Rewired `PathConfig`/`PathChip` to use swap chip; simplified `calculate_root` to a swap+hash loop |

### Impact

| Height | Old overhead rows | New overhead rows | Rows saved |
|--------|-------------------|-------------------|------------|
| 3      | 9                 | 3                 | 6          |
| 10     | 30                | 10                | 20         |
| 20     | 60                | 20                | 40         |

The Poseidon hash (~64 rows/level) still dominates, so this is a ~3% reduction in total rows at height 20. The primary benefits are reduced circuit complexity (3 constraint polynomials per level instead of 6), and potentially allowing a smaller `k` at boundary cases, which halves the SRS size.

## Variable-Length Sparse Path Optimization

**Goal:** Reduce the number of Poseidon hashes in the circuit from N (tree height) to MAX_K (an upper bound on non-zero sibling levels), cutting prover time by ~30% for sparse trees.

### Problem

The dense `PathChip` always iterates over all N levels of the tree, computing N Poseidon hashes (~68 rows each). In a sparse tree (e.g., 100M items in a height-53 tree), only K ~ 27 levels have siblings that differ from the precomputed empty hash. The remaining ~26 levels hash against known empty values -- work the circuit doesn't need to do.

### Approach

The verification is split into two parts:

1. **Circuit (K hashes):** A new `SparsePathChip<..., MAX_K>` iterates MAX_K times. Each iteration does `swap + poseidon_hash + conditional_select`. Active slots (real siblings) use the hash result; inactive slots (padding) pass through the previous hash unchanged via conditional select. The circuit proves: `leaf` hashes to `compact_root` through the K non-zero levels.

2. **Native gap verification (free):** Outside the circuit, the caller verifies that `compact_root` chains through the N-K empty-hash gap levels to produce the standard SMT root. This is deterministic and doesn't need a proof.

```
leaf ──[K non-zero hashes, in circuit]──> compact root
compact root ──[N-K gap hashes, native]──> standard SMT root
```

### How conditional_select handles padding

PLONKish circuits have a fixed shape, so MAX_K iterations always run. An `is_active` flag per slot controls whether the hash result is used:

- `is_active = 1`: `out = hash_result` (real entry, advance the hash chain)
- `is_active = 0`: `out = prev` (padding, pass through unchanged)

The constraint `cond * (1 - cond) = 0` enforces `is_active` is boolean, preventing a malicious prover from partially blending results.

### Files Changed

| File | Change |
|------|--------|
| `smt/src/smt.rs` | Added `SparsePathEntry`, `SparsePath` types with `calculate_compact_root()`, `calculate_full_root()`, `to_padded_arrays()`. Added `generate_sparse_membership_proof()` to `SparseMerkleTree`. |
| `chiplet/src/smt_chip.rs` | Added `SparsePathConfig`/`SparsePathChip` with `configure()`, `from_native()`, `calculate_root()` (swap + hash + conditional_select loop), `check_membership()`. |

### Choosing MAX_K

MAX_K is a const generic (fixed at circuit setup time). Set it to `ceil(log2(max_items)) + 4`:

| Expected max items | K upper bound | Recommended MAX_K |
|---|---|---|
| ~1,000 | ~10 | 16 |
| ~1,000,000 | ~20 | 24 |
| ~100,000,000 | ~27 | 32 |
| ~1 billion | ~30 | 34 |

If MAX_K is too small, `to_padded_arrays` panics at proof time. If too large, extra padding slots waste prover time (~70 rows each) but are otherwise harmless.

### Impact (N=53, 100M items, K=27, MAX_K=32, k=12)

| Metric | Dense (N=53) | Sparse (MAX_K=32) | Improvement |
|--------|---|---|---|
| Poseidon hashes | 53 | 32 | 40% fewer |
| Circuit rows | ~3,700 | ~2,200 | 40% fewer |
| create_proof | 1.94s | 1.32s | 32% faster |
| keygen_vk | 1.30s | 0.90s | 31% faster |
| verify_proof | 25ms | 18ms | 28% faster |
| Params size (k=12) | 0.25 MB | 0.25 MB | same |

Both circuits fit in k=12. The savings come from fewer Poseidon hashes during proving, not from reducing k. The SRS size and polynomial degree are identical. The dense `PathChip` is preserved for backward compatibility.
