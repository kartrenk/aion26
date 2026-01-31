# Flop Abstraction Implementation Plan

## Overview

Implement two poker AI optimizations to reduce the effective game tree:
1. **Flop Bucketing** - K-Means clustering on equity histograms
2. **Flop Subsets** - Train on 184 representative flops

## Current State

- **Suit Isomorphism**: ✅ Implemented (22,100 → 1,755 distinct flops)
- **Bucketing**: ❌ Not implemented
- **Flop Subsets**: ❌ Not implemented

## Phase 1: Equity Histogram Infrastructure

### 1.1 Precompute All 1,755 Canonical Flops

```rust
// src/aion26_rust/src/flop_abstraction.rs

/// Generate all 1,755 strategically distinct flops
/// Uses suit isomorphism: only generate canonical suit patterns
pub fn generate_canonical_flops() -> Vec<[u8; 3]> {
    // Patterns: rainbow (3 suits), two-tone (2 suits), monotone (1 suit)
    // For each rank combo, generate one representative per pattern
}
```

### 1.2 Equity Histogram Calculation

For each flop, compute how it affects a standard preflop range:

```rust
/// Calculate equity histogram for a flop
/// Returns: [f32; 50] - equity distribution in 2% buckets
pub fn compute_equity_histogram(flop: [u8; 3], range: &[(u8, u8)]) -> [f32; 50] {
    // For each hand in range:
    //   1. Remove blocked cards
    //   2. Enumerate all possible opponent hands
    //   3. Calculate equity vs opponent range
    //   4. Bucket into histogram
}
```

### 1.3 Precompute and Cache

```rust
/// Precompute equity histograms for all 1,755 flops
/// Store in binary file for fast loading
pub fn precompute_all_histograms() -> HashMap<[u8; 3], [f32; 50]> {
    // ~1,755 flops × 1,326 hands × enumeration
    // Cache to disk: flop_histograms.bin
}
```

## Phase 2: K-Means Flop Bucketing

### 2.1 Clustering Algorithm

```rust
// src/aion26_rust/src/flop_clustering.rs

/// K-Means clustering on equity histograms
/// Groups similar flops into buckets
pub struct FlopBucketing {
    num_buckets: usize,           // e.g., 200 buckets
    centroids: Vec<[f32; 50]>,    // Cluster centers
    assignments: HashMap<[u8; 3], usize>,  // Flop → bucket ID
}

impl FlopBucketing {
    /// Run K-Means clustering
    pub fn fit(histograms: &HashMap<[u8; 3], [f32; 50]>, k: usize) -> Self {
        // 1. Initialize k centroids (k-means++)
        // 2. Iterate until convergence:
        //    a. Assign each flop to nearest centroid (EMD or L2)
        //    b. Update centroids as mean of assigned flops
        // 3. Store assignments
    }

    /// Get bucket ID for a flop
    pub fn get_bucket(&self, flop: [u8; 3]) -> usize {
        self.assignments[&flop]
    }
}
```

### 2.2 Earth Mover's Distance (EMD)

Better than L2 for comparing distributions:

```rust
/// Earth Mover's Distance between two histograms
pub fn emd(h1: &[f32; 50], h2: &[f32; 50]) -> f32 {
    // Cumulative difference metric
    let mut emd = 0.0;
    let mut cumsum = 0.0;
    for i in 0..50 {
        cumsum += h1[i] - h2[i];
        emd += cumsum.abs();
    }
    emd
}
```

## Phase 3: Flop Subset Selection

### 3.1 Representative Flop Selection

Select 184 flops that minimize approximation error:

```rust
// src/aion26_rust/src/flop_subsets.rs

/// Select representative flops using greedy coverage
pub fn select_representative_flops(
    histograms: &HashMap<[u8; 3], [f32; 50]>,
    n: usize,  // e.g., 184
) -> Vec<[u8; 3]> {
    // Greedy algorithm:
    // 1. Start with empty set
    // 2. Repeatedly add flop that most reduces max-error
    // 3. Stop when n flops selected
}
```

### 3.2 Flop Mapping

Map any flop to its nearest representative:

```rust
/// Map a flop to its nearest representative
pub fn map_to_representative(
    flop: [u8; 3],
    representatives: &[[u8; 3]],
    histograms: &HashMap<[u8; 3], [f32; 50]>,
) -> [u8; 3] {
    // Find representative with minimum EMD
}
```

## Phase 4: Integration into Training

### 4.1 Modified State Encoding

```rust
fn encode_state(state: &RustFullHoldem, player: u8) -> Vec<f32> {
    // ... existing code ...

    // NEW: Use bucket ID instead of raw board encoding
    if GAME_MODE == "full" && USE_FLOP_BUCKETS {
        let canonical_flop = canonicalize_flop(&state.board[0..3]);
        let bucket_id = FLOP_BUCKETING.get_bucket(canonical_flop);

        // Encode bucket_id as one-hot or embedding
        // This replaces the 85-dim board encoding
    }
}
```

### 4.2 Modified Traversal (Subset Mode)

```rust
impl ParallelTrainerFull {
    fn start_traversal(&mut self) {
        // NEW: Only sample from representative flops
        if USE_FLOP_SUBSETS {
            let flop = REPRESENTATIVE_FLOPS[rng.gen_range(0..184)];
            // Deal this specific flop instead of random
        }
    }
}
```

## Phase 5: Configuration

```rust
// src/aion26_rust/src/config.rs

pub struct AbstractionConfig {
    pub use_suit_isomorphism: bool,  // Already implemented
    pub use_flop_buckets: bool,
    pub num_buckets: usize,          // e.g., 200
    pub use_flop_subsets: bool,
    pub num_subsets: usize,          // e.g., 184
}
```

## File Structure

```
src/aion26_rust/src/
├── flop_abstraction.rs    # Canonical flop generation
├── equity_calculator.rs   # Equity histogram computation
├── flop_clustering.rs     # K-Means bucketing
├── flop_subsets.rs        # Representative selection
├── abstraction_config.rs  # Configuration
└── lib.rs                 # Export new modules
```

## Implementation Order

1. **equity_calculator.rs** - Core equity calculation
2. **flop_abstraction.rs** - Generate 1,755 canonical flops
3. **flop_clustering.rs** - K-Means with EMD
4. **flop_subsets.rs** - Greedy subset selection
5. **Integration** - Modify encode_state and traversal
6. **Testing** - Validate approximation error

## Expected Results

| Stage | Distinct States | Reduction |
|-------|-----------------|-----------|
| Raw | 22,100 flops | 1x |
| Suit Isomorphism | 1,755 flops | 12.5x |
| Bucketing (200) | 200 buckets | 110x |
| Subsets (184) | 184 flops | 120x |
| Combined | ~150-200 | **~110-150x** |

## Risks and Mitigations

1. **Approximation Error**: Start with more buckets (200+), tune down
2. **Precomputation Time**: Cache histograms to disk
3. **Memory**: Lazy-load bucket assignments
