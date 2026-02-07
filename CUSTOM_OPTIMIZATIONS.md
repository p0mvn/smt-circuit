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

**Proof generation** walks the compressed tree from root to the target leaf. At `Multi` nodes, it picks the target child and records the sibling's precomputed hash. At `Single`/`Double` nodes, it fills in remaining levels with empty hashes (or the other leaf's hash at the divergence point). The resulting `Path` and `SparsePath` are identical to those produced by `SparseMerkleTree`.

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

    // Sparse proof — only K non-empty levels. Compatible with SparsePathChip.
    pub fn generate_sparse_membership_proof(&self, index: u64) -> SparsePath<F, H>;
}
```

### Files changed

| File | Change |
|------|--------|
| `smt/src/compressed_smt.rs` | New file: `CompressedNode` enum, `CompressedSMT` struct, build/hash/proof functions |
| `smt/src/lib.rs` | Added `pub mod compressed_smt;` |
| `smt/Cargo.toml` | Added `rayon = "1"` dependency |

No changes to `chiplet/` — the compressed SMT produces the same `Path` and `SparsePath` types that the circuit chips already consume.

### Test coverage

| Category | Tests | What is verified |
|----------|-------|------------------|
| Root correctness | 7 | Identical root hash vs `SparseMerkleTree` for heights 3/10/20, 0-1000 leaves, sparse indices |
| Node classification | 1 | Zero/Single/Double/Multi for 0/1/2/3 leaves |
| Dense proof | 3 | Bit-exact path match vs original, membership check, sparse-index proofs |
| Sparse proof | 3 | Entry-exact match vs original, full root reconstruction, compact root |
| Scale | 2 | 1K leaves at height 20, 10K random leaves at height 53 |
