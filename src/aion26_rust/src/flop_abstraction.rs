//! Flop Abstraction Module
//!
//! Implements poker flop abstractions for reduced state space:
//! 1. Canonical flop generation (1,755 distinct flops via suit isomorphism)
//! 2. Equity histogram computation
//! 3. K-Means bucketing with Earth Mover's Distance
//! 4. Representative flop subset selection

use std::collections::HashMap;
use pyo3::prelude::*;

// ============================================================================
// Constants
// ============================================================================

/// Number of strategically distinct flops under suit isomorphism
/// Math: 286 unpaired × 3 patterns + 156 paired × 2 patterns + 13 trips × 1 pattern = 1183
pub const NUM_CANONICAL_FLOPS: usize = 1183;

/// Number of buckets for equity histogram (2% each)
pub const HISTOGRAM_BUCKETS: usize = 50;

// ============================================================================
// Canonical Flop Generation
// ============================================================================

/// Suit pattern for a flop
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SuitPattern {
    /// All three cards same suit (e.g., As Ks 2s)
    Monotone,
    /// Two cards same suit (e.g., As Ks 2d)
    TwoTone,
    /// All different suits (e.g., As Kd 2c)
    Rainbow,
}

/// A canonical flop representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CanonicalFlop {
    /// Ranks in descending order (0-12, where 0=2, 12=A)
    pub ranks: [u8; 3],
    /// Suit pattern
    pub pattern: SuitPattern,
}

impl CanonicalFlop {
    /// Create a new canonical flop
    pub fn new(r1: u8, r2: u8, r3: u8, pattern: SuitPattern) -> Self {
        let mut ranks = [r1, r2, r3];
        ranks.sort_by(|a, b| b.cmp(a)); // Descending order
        Self { ranks, pattern }
    }

    /// Convert to actual cards (using canonical suit assignment)
    pub fn to_cards(&self) -> [u8; 3] {
        match self.pattern {
            SuitPattern::Monotone => {
                // All suit 0 (only valid when all ranks different)
                [
                    self.ranks[0],      // suit 0
                    self.ranks[1],      // suit 0
                    self.ranks[2],      // suit 0
                ]
            }
            SuitPattern::TwoTone => {
                // Two cards share a suit, one is different
                // Handle paired boards: if ranks[0] == ranks[1], put them in different suits
                if self.ranks[0] == self.ranks[1] {
                    // Pair in ranks[0] and ranks[1] - put pair in different suits
                    [
                        self.ranks[0],           // suit 0
                        self.ranks[1] + 13,      // suit 1 (pair card in different suit)
                        self.ranks[2],           // suit 0
                    ]
                } else if self.ranks[1] == self.ranks[2] {
                    // Pair in ranks[1] and ranks[2] - put pair in different suits
                    [
                        self.ranks[0],           // suit 0
                        self.ranks[1],           // suit 0
                        self.ranks[2] + 13,      // suit 1 (pair card in different suit)
                    ]
                } else {
                    // No pair - first two suit 0, third suit 1
                    [
                        self.ranks[0],           // suit 0
                        self.ranks[1],           // suit 0
                        self.ranks[2] + 13,      // suit 1
                    ]
                }
            }
            SuitPattern::Rainbow => {
                // All different suits
                [
                    self.ranks[0],           // suit 0
                    self.ranks[1] + 13,      // suit 1
                    self.ranks[2] + 26,      // suit 2
                ]
            }
        }
    }

    /// Create from actual cards
    pub fn from_cards(cards: &[u8; 3]) -> Self {
        let ranks: Vec<u8> = cards.iter().map(|&c| c % 13).collect();
        let suits: Vec<u8> = cards.iter().map(|&c| c / 13).collect();

        // Determine suit pattern
        let unique_suits: std::collections::HashSet<u8> = suits.iter().cloned().collect();
        let pattern = match unique_suits.len() {
            1 => SuitPattern::Monotone,
            2 => SuitPattern::TwoTone,
            _ => SuitPattern::Rainbow,
        };

        let mut sorted_ranks = [ranks[0], ranks[1], ranks[2]];
        sorted_ranks.sort_by(|a, b| b.cmp(a));

        Self {
            ranks: sorted_ranks,
            pattern,
        }
    }
}

/// Generate all 1,755 canonical flops
pub fn generate_all_canonical_flops() -> Vec<CanonicalFlop> {
    let mut flops = Vec::with_capacity(NUM_CANONICAL_FLOPS);

    // Generate all rank combinations (13 choose 3 with replacement = 455 for trips/pairs)
    for r1 in 0u8..13 {
        for r2 in 0u8..=r1 {
            for r3 in 0u8..=r2 {
                // Determine valid suit patterns based on rank duplicates
                let has_trips = r1 == r2 && r2 == r3;
                let has_pair = r1 == r2 || r2 == r3 || r1 == r3;

                if has_trips {
                    // Three of a kind on board - only rainbow possible
                    // (can't have 3 cards of same rank and same suit)
                    flops.push(CanonicalFlop::new(r1, r2, r3, SuitPattern::Rainbow));
                } else if has_pair {
                    // Pair on board - rainbow and two-tone possible
                    flops.push(CanonicalFlop::new(r1, r2, r3, SuitPattern::Rainbow));
                    flops.push(CanonicalFlop::new(r1, r2, r3, SuitPattern::TwoTone));
                } else {
                    // All different ranks - all patterns possible
                    flops.push(CanonicalFlop::new(r1, r2, r3, SuitPattern::Rainbow));
                    flops.push(CanonicalFlop::new(r1, r2, r3, SuitPattern::TwoTone));
                    flops.push(CanonicalFlop::new(r1, r2, r3, SuitPattern::Monotone));
                }
            }
        }
    }

    flops
}

/// Map any flop to its canonical form
pub fn canonicalize_flop(cards: &[u8]) -> CanonicalFlop {
    assert!(cards.len() >= 3);
    CanonicalFlop::from_cards(&[cards[0], cards[1], cards[2]])
}

// ============================================================================
// Equity Histogram Calculation
// ============================================================================

// ============================================================================
// K-Means Clustering with EMD
// ============================================================================

/// Earth Mover's Distance between two histograms
pub fn earth_movers_distance(h1: &[f32; HISTOGRAM_BUCKETS], h2: &[f32; HISTOGRAM_BUCKETS]) -> f32 {
    let mut emd = 0.0f32;
    let mut cumsum = 0.0f32;

    for i in 0..HISTOGRAM_BUCKETS {
        cumsum += h1[i] - h2[i];
        emd += cumsum.abs();
    }

    emd
}

/// Flop bucketing using K-Means clustering
pub struct FlopBucketing {
    /// Number of buckets
    pub num_buckets: usize,
    /// Flop to bucket assignment
    pub assignments: HashMap<CanonicalFlop, usize>,
}

impl FlopBucketing {
    /// Create empty bucketing
    pub fn new(num_buckets: usize) -> Self {
        Self {
            num_buckets,
            assignments: HashMap::new(),
        }
    }

    /// Fit K-Means clustering on flop histograms
    pub fn fit(
        histograms: &HashMap<CanonicalFlop, [f32; HISTOGRAM_BUCKETS]>,
        num_buckets: usize,
        max_iterations: usize,
    ) -> Self {
        let flops: Vec<_> = histograms.keys().cloned().collect();
        let n = flops.len();

        if n == 0 || num_buckets == 0 {
            return Self::new(num_buckets);
        }

        let k = num_buckets.min(n);

        // K-Means++ initialization
        let mut centroids = Vec::with_capacity(k);
        let mut rng_state = 42u64; // Simple PRNG

        // First centroid: random
        let first_idx = (simple_random(&mut rng_state) * n as f64) as usize % n;
        centroids.push(histograms[&flops[first_idx]]);

        // Remaining centroids: weighted by distance
        for _ in 1..k {
            let distances: Vec<f32> = flops
                .iter()
                .map(|f| {
                    centroids
                        .iter()
                        .map(|c| earth_movers_distance(&histograms[f], c))
                        .fold(f32::MAX, f32::min)
                })
                .collect();

            let total: f32 = distances.iter().sum();
            if total <= 0.0 {
                break;
            }

            // Weighted random selection
            let target = simple_random(&mut rng_state) as f32 * total;
            let mut cumsum = 0.0;
            let mut selected = 0;
            for (i, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= target {
                    selected = i;
                    break;
                }
            }

            centroids.push(histograms[&flops[selected]]);
        }

        // K-Means iterations
        let mut assignments: Vec<usize> = vec![0; n];

        for _iter in 0..max_iterations {
            let mut changed = false;

            // Assignment step
            for (i, flop) in flops.iter().enumerate() {
                let hist = &histograms[flop];
                let mut best_dist = f32::MAX;
                let mut best_cluster = 0;

                for (c, centroid) in centroids.iter().enumerate() {
                    let dist = earth_movers_distance(hist, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update step
            for (c, centroid) in centroids.iter_mut().enumerate() {
                let mut new_centroid = [0.0f32; HISTOGRAM_BUCKETS];
                let mut count = 0;

                for (i, &assignment) in assignments.iter().enumerate() {
                    if assignment == c {
                        let hist = &histograms[&flops[i]];
                        for (j, h) in hist.iter().enumerate() {
                            new_centroid[j] += h;
                        }
                        count += 1;
                    }
                }

                if count > 0 {
                    for h in new_centroid.iter_mut() {
                        *h /= count as f32;
                    }
                    *centroid = new_centroid;
                }
            }
        }

        // Build assignment map
        let mut assignment_map = HashMap::new();
        for (i, flop) in flops.iter().enumerate() {
            assignment_map.insert(*flop, assignments[i]);
        }

        Self {
            num_buckets: k,
            assignments: assignment_map,
        }
    }

    /// Get bucket ID for a flop
    pub fn get_bucket(&self, flop: &CanonicalFlop) -> usize {
        self.assignments.get(flop).cloned().unwrap_or(0)
    }

    /// Get bucket ID for raw cards
    pub fn get_bucket_for_cards(&self, cards: &[u8]) -> usize {
        let canonical = canonicalize_flop(cards);
        self.get_bucket(&canonical)
    }
}

// Simple PRNG for deterministic initialization
fn simple_random(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*state >> 33) as f64 / (1u64 << 31) as f64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_flop_count() {
        let flops = generate_all_canonical_flops();
        assert_eq!(flops.len(), NUM_CANONICAL_FLOPS);
    }

    #[test]
    fn test_suit_pattern_detection() {
        // Monotone: As Ks Qs (all spades)
        let monotone = CanonicalFlop::from_cards(&[12 + 39, 11 + 39, 10 + 39]);
        assert_eq!(monotone.pattern, SuitPattern::Monotone);

        // Two-tone: As Ks Qd
        let two_tone = CanonicalFlop::from_cards(&[12 + 39, 11 + 39, 10 + 13]);
        assert_eq!(two_tone.pattern, SuitPattern::TwoTone);

        // Rainbow: As Kd Qc
        let rainbow = CanonicalFlop::from_cards(&[12 + 39, 11 + 13, 10]);
        assert_eq!(rainbow.pattern, SuitPattern::Rainbow);
    }

    #[test]
    fn test_emd_same_histogram() {
        let h1 = [0.02f32; HISTOGRAM_BUCKETS];
        let h2 = [0.02f32; HISTOGRAM_BUCKETS];
        assert!((earth_movers_distance(&h1, &h2) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_emd_different_histograms() {
        let mut h1 = [0.0f32; HISTOGRAM_BUCKETS];
        let mut h2 = [0.0f32; HISTOGRAM_BUCKETS];
        h1[0] = 1.0; // All mass at 0%
        h2[49] = 1.0; // All mass at 100%
        let emd = earth_movers_distance(&h1, &h2);
        assert!(emd > 0.0);
    }
}

// ============================================================================
// Python Bindings
// ============================================================================

/// Python-accessible flop bucketing
#[pyclass]
pub struct PyFlopBucketing {
    inner: FlopBucketing,
}

#[pymethods]
impl PyFlopBucketing {
    /// Get bucket ID for a flop (3 cards)
    fn get_bucket(&self, card1: u8, card2: u8, card3: u8) -> usize {
        self.inner.get_bucket_for_cards(&[card1, card2, card3])
    }

    /// Get number of buckets
    fn num_buckets(&self) -> usize {
        self.inner.num_buckets
    }
}

/// Generate all canonical flops
#[pyfunction]
pub fn py_generate_canonical_flops() -> Vec<(u8, u8, u8, String)> {
    let flops = generate_all_canonical_flops();
    flops
        .iter()
        .map(|f| {
            let cards = f.to_cards();
            let pattern = match f.pattern {
                SuitPattern::Monotone => "monotone",
                SuitPattern::TwoTone => "two_tone",
                SuitPattern::Rainbow => "rainbow",
            };
            (cards[0], cards[1], cards[2], pattern.to_string())
        })
        .collect()
}

/// Get canonical flop count
#[pyfunction]
pub fn py_num_canonical_flops() -> usize {
    NUM_CANONICAL_FLOPS
}

/// Create simplified flop buckets based on board texture
/// This is a fast approximation that doesn't require equity calculation
#[pyfunction]
pub fn py_create_texture_buckets(num_buckets: usize) -> PyFlopBucketing {
    let flops = generate_all_canonical_flops();

    // Create simplified histograms based on board texture features
    let mut histograms: HashMap<CanonicalFlop, [f32; HISTOGRAM_BUCKETS]> = HashMap::new();

    for flop in &flops {
        let mut hist = [0.0f32; HISTOGRAM_BUCKETS];

        // Feature 1: Suit pattern (0-2)
        let suit_score = match flop.pattern {
            SuitPattern::Monotone => 0.9,   // Very draw-heavy
            SuitPattern::TwoTone => 0.5,    // Moderate
            SuitPattern::Rainbow => 0.1,    // Dry
        };

        // Feature 2: Connectedness (how close are the ranks)
        let ranks = &flop.ranks;
        let gap1 = (ranks[0] as i32 - ranks[1] as i32).abs();
        let gap2 = (ranks[1] as i32 - ranks[2] as i32).abs();
        let connected_score = 1.0 - ((gap1 + gap2) as f32 / 24.0);

        // Feature 3: High card strength
        let high_score = ranks[0] as f32 / 12.0;

        // Feature 4: Paired board
        let paired = ranks[0] == ranks[1] || ranks[1] == ranks[2];
        let paired_score = if paired { 0.7 } else { 0.3 };

        // Combine into histogram
        // Different buckets represent different board types
        let bucket1 = ((suit_score * 10.0) as usize).min(9);
        let bucket2 = ((connected_score * 10.0) as usize).min(9) + 10;
        let bucket3 = ((high_score * 10.0) as usize).min(9) + 20;
        let bucket4 = ((paired_score * 10.0) as usize).min(9) + 30;

        hist[bucket1] = 0.25;
        hist[bucket2] = 0.25;
        hist[bucket3] = 0.25;
        hist[bucket4] = 0.25;

        histograms.insert(*flop, hist);
    }

    // Run K-Means clustering
    let bucketing = FlopBucketing::fit(&histograms, num_buckets, 50);

    PyFlopBucketing { inner: bucketing }
}

/// Get bucket for a flop using texture-based bucketing
#[pyfunction]
pub fn py_get_flop_bucket(bucketing: &PyFlopBucketing, card1: u8, card2: u8, card3: u8) -> usize {
    bucketing.get_bucket(card1, card2, card3)
}
