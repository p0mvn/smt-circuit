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

## Compressed Sparse Merkle Tree

**Goal:** Replace the `BTreeMap`-based `SparseMerkleTree` with a compressed node tree that reduces memory from ~80 GB to ~6 GB for 51.7M leaves at height 53, and enables multi-core parallel construction via rayon.

### Problem

The `SparseMerkleTree` stores every internal node in a `BTreeMap<u64, F>`. For 51.7M leaves at height 53, this creates ~1.4 billion entries consuming ~80-90 GB of RAM. The machine swaps to disk, making construction take 10+ hours.

### Approach

Instead of materializing every internal node, classify each subtree by how many non-default leaves it contains:

| Variant | Condition | Storage | Hash |
|---------|-----------|---------|------|
| `Zero` | 0 leaves in subtree | 8 bytes (tag only) | Looked up from `empty_hashes[level]` |
| `Single` | 1 leaf | 48 bytes (index + value + hash) | Leaf hashed up through empty siblings |
| `Double` | 2 leaves | 88 bytes (two pairs + hash) | Two Single paths merged at divergence point |
| `Multi` | 3+ leaves | 56 bytes (hash + 2 Box ptrs) | `poseidon(left.hash, right.hash)` — only variant that recurses |

**Construction** is top-down recursive. The `build()` function splits sorted leaves by the current bit position and recurses into left/right children. For subtrees with > 1024 leaves, `rayon::join` runs the two children in parallel. Hashes for `Single` and `Double` nodes are computed eagerly during construction.

**Proof generation** walks the compressed tree from root to the target leaf. At `Multi` nodes, it picks the target child and records the sibling's precomputed hash. At `Single`/`Double` nodes, it fills in remaining levels with empty hashes (or the other leaf's hash at the divergence point). The resulting `Path` is identical to those produced by `SparseMerkleTree`.

### Hash helpers

**`compute_single_hash`** — hashes one leaf up through `level` layers of empty siblings:

```
h = leaf_value
for l in 0..level:
    bit = (leaf_index >> l) & 1
    h = hash(h, empty[l])  if bit == 0
    h = hash(empty[l], h)  if bit == 1
```

**`compute_double_hash`** — finds the divergence point of two leaves, hashes each up to that point, merges, then continues up through empty siblings:

```
div_level = highest_differing_bit(a, b) + 1
a_hash = single_hash(a, div_level - 1)
b_hash = single_hash(b, div_level - 1)
h = hash(a_hash, b_hash)  [ordered by bit at div_level - 1]
for l in div_level..level:
    h = hash(h, empty[l]) or hash(empty[l], h)
```

### Memory analysis (51.7M leaves, N=53)

Leaves are ~0.0006% dense in the 2^53 address space. Collisions begin around level ~27 from the leaf level.

- Total compressed nodes: ~103M (≈ 2 × 51.7M)
- Average node size: ~56 bytes
- **Total memory: ~6 GB** (vs ~80-90 GB with `BTreeMap`)

### Hash count analysis

The total Poseidon hash count is ~1.5 billion regardless of data structure. The compressed tree does **not** reduce the total hash count — the win is purely that everything fits in RAM, eliminating swap thrashing. With rayon parallelism:

| Cores | Estimated time (at 28μs/hash) |
|-------|-------------------------------|
| 1     | ~11 hours                     |
| 8     | ~1.4 hours                    |
| 16    | ~42 minutes                   |
| 32    | ~21 minutes                   |

### Public API

```rust
impl<F, H, const N: usize> CompressedSMT<F, H, N> {
    // Build from u64-indexed leaves (full 2^N address space). H: Sync for rayon.
    pub fn new(leaves: &BTreeMap<u64, F>, hasher: &H, empty_leaf: &[u8; 64]) -> Result<Self>;

    // Backward-compatible constructor from u32-indexed leaves.
    pub fn new_from_u32(leaves: &BTreeMap<u32, F>, hasher: &H, empty_leaf: &[u8; 64]) -> Result<Self>;

    // Root hash (matches SparseMerkleTree::root() for identical inputs).
    pub fn root(&self) -> F;

    // Dense proof — all N levels. Compatible with PathChip.
    pub fn generate_membership_proof(&self, index: u64) -> Path<F, H, N>;
}
```

### Files changed

| File | Change |
|------|--------|
| `smt/src/compressed_smt.rs` | New file: `CompressedNode` enum, `CompressedSMT` struct, build/hash/proof functions |
| `smt/src/lib.rs` | Added `pub mod compressed_smt;` |
| `smt/Cargo.toml` | Added `rayon = "1"` dependency |

No changes to `chiplet/` — the compressed SMT produces the same `Path` type that `PathChip` already consumes.

### Test coverage

| Category | Tests | What is verified |
|----------|-------|------------------|
| Root correctness | 7 | Identical root hash vs `SparseMerkleTree` for heights 3/10/20, 0-1000 leaves, sparse indices |
| Node classification | 1 | Zero/Single/Double/Multi for 0/1/2/3 leaves |
| Dense proof | 3 | Bit-exact path match vs original, membership check, sparse-index proofs |
| Scale | 2 | 1K leaves at height 20, 10K random leaves at height 53 |
