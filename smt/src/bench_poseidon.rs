/// Benchmarks comparing Poseidon (old) vs Poseidon2 (new) hash performance.
///
/// Run with: cargo test -p smt bench_poseidon -- --nocapture --ignored
#[cfg(test)]
mod tests {
    use crate::poseidon::{FieldHasher, Poseidon};
    use crate::poseidon2::Poseidon2;
    use crate::smt::SparseMerkleTree;
    use ff::Field;
    use pasta_curves::Fp;
    use rand::rngs::OsRng;
    use std::collections::BTreeMap;
    use std::time::Instant;

    /// Benchmark raw hash throughput: Poseidon1 vs Poseidon2
    #[test]
    #[ignore]
    fn bench_hash_throughput() {
        let rng = OsRng;
        let iterations = 10_000;

        // Prepare random inputs
        let inputs: Vec<[Fp; 2]> = (0..iterations)
            .map(|_| [Fp::random(rng), Fp::random(rng)])
            .collect();

        // --- Poseidon 1 (old) ---
        let hasher1 = Poseidon::<Fp, 2>::new();
        let start = Instant::now();
        for inp in &inputs {
            let _ = hasher1.hash(*inp).unwrap();
        }
        let poseidon1_time = start.elapsed();

        // --- Poseidon 2 (new) ---
        let hasher2 = Poseidon2::<Fp, 2>::new();
        let start = Instant::now();
        for inp in &inputs {
            let _ = hasher2.hash(*inp).unwrap();
        }
        let poseidon2_time = start.elapsed();

        let speedup = poseidon1_time.as_secs_f64() / poseidon2_time.as_secs_f64();

        println!("\n========================================");
        println!("  RAW HASH BENCHMARK ({} iterations)", iterations);
        println!("========================================");
        println!(
            "  Poseidon1 (old): {:>10.3} ms  ({:.1} us/hash)",
            poseidon1_time.as_secs_f64() * 1000.0,
            poseidon1_time.as_secs_f64() * 1_000_000.0 / iterations as f64
        );
        println!(
            "  Poseidon2 (new): {:>10.3} ms  ({:.1} us/hash)",
            poseidon2_time.as_secs_f64() * 1000.0,
            poseidon2_time.as_secs_f64() * 1_000_000.0 / iterations as f64
        );
        println!(
            "  Speedup:         {:.2}x {}",
            if speedup >= 1.0 { speedup } else { 1.0 / speedup },
            if speedup >= 1.0 {
                "(Poseidon2 is faster)"
            } else {
                "(Poseidon1 is faster)"
            }
        );
        println!("========================================\n");
    }

    /// Benchmark SMT tree construction: small tree (height 20, ~1K leaves)
    #[test]
    #[ignore]
    fn bench_smt_small_tree() {
        bench_smt_construction::<20>(1_000, "SMALL TREE (height=20, 1K leaves)");
    }

    /// Benchmark SMT tree construction: medium tree (height 20, ~10K leaves)
    #[test]
    #[ignore]
    fn bench_smt_medium_tree() {
        bench_smt_construction::<20>(10_000, "MEDIUM TREE (height=20, 10K leaves)");
    }

    /// Benchmark SMT tree construction: larger tree (height 20, ~100K leaves)
    #[test]
    #[ignore]
    fn bench_smt_large_tree() {
        bench_smt_construction::<20>(100_000, "LARGE TREE (height=20, 100K leaves)");
    }

    /// Benchmark SMT tree construction at depth 53 with a small number of leaves
    /// to estimate per-leaf cost at production depth.
    #[test]
    #[ignore]
    fn bench_smt_depth53_sparse() {
        bench_smt_construction::<53>(1_000, "DEPTH-53 SPARSE (height=53, 1K leaves)");
    }

    /// Benchmark SMT tree construction at depth 53 with 10K leaves.
    #[test]
    #[ignore]
    fn bench_smt_depth53_10k() {
        bench_smt_construction::<53>(10_000, "DEPTH-53 (height=53, 10K leaves)");
    }

    fn bench_smt_construction<const N: usize>(num_leaves: u32, label: &str) {
        let rng = OsRng;
        let empty_leaf = [0u8; 64];

        // Generate random leaves
        let leaves: BTreeMap<u32, Fp> = (0..num_leaves)
            .map(|i| (i, Fp::random(rng)))
            .collect();

        // --- Poseidon 1 (old) ---
        let hasher1 = Poseidon::<Fp, 2>::new();
        let start = Instant::now();
        let smt1 = SparseMerkleTree::<Fp, Poseidon<Fp, 2>, N>::new(
            &leaves, &hasher1, &empty_leaf,
        )
        .unwrap();
        let poseidon1_time = start.elapsed();
        let root1 = smt1.root();

        // --- Poseidon 2 (new) ---
        let hasher2 = Poseidon2::<Fp, 2>::new();
        let start = Instant::now();
        let smt2 = SparseMerkleTree::<Fp, Poseidon2<Fp, 2>, N>::new(
            &leaves, &hasher2, &empty_leaf,
        )
        .unwrap();
        let poseidon2_time = start.elapsed();
        let root2 = smt2.root();

        let speedup = poseidon1_time.as_secs_f64() / poseidon2_time.as_secs_f64();

        println!("\n========================================");
        println!("  SMT BENCHMARK: {}", label);
        println!("========================================");
        println!(
            "  Poseidon1 (old): {:>10.3} ms",
            poseidon1_time.as_secs_f64() * 1000.0
        );
        println!(
            "  Poseidon2 (new): {:>10.3} ms",
            poseidon2_time.as_secs_f64() * 1000.0
        );
        println!(
            "  Speedup:         {:.2}x {}",
            if speedup >= 1.0 { speedup } else { 1.0 / speedup },
            if speedup >= 1.0 {
                "(Poseidon2 is faster)"
            } else {
                "(Poseidon1 is faster)"
            }
        );
        println!("  Roots match:     {}", root1 != root2); // Should differ (different hash functions)
        println!(
            "  Per-leaf cost:   P1={:.1} us, P2={:.1} us",
            poseidon1_time.as_secs_f64() * 1_000_000.0 / num_leaves as f64,
            poseidon2_time.as_secs_f64() * 1_000_000.0 / num_leaves as f64
        );

        // Extrapolate to 51M leaves
        let p1_per_leaf_us = poseidon1_time.as_secs_f64() * 1_000_000.0 / num_leaves as f64;
        let p2_per_leaf_us = poseidon2_time.as_secs_f64() * 1_000_000.0 / num_leaves as f64;
        let target_leaves = 51_000_000.0;
        println!("  ---");
        println!(
            "  Estimated 51M leaves:  P1={:.1} min, P2={:.1} min",
            p1_per_leaf_us * target_leaves / 1_000_000.0 / 60.0,
            p2_per_leaf_us * target_leaves / 1_000_000.0 / 60.0
        );
        println!("========================================\n");
    }
}
