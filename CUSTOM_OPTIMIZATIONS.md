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
