/// Fast 7-card poker hand evaluator using lookup tables
///
/// Cards are represented as u8 (0-51):
/// - Rank: card % 13 (0=2, 1=3, ..., 12=A)
/// - Suit: card / 13 (0=Spades, 1=Hearts, 2=Diamonds, 3=Clubs)
///
/// Returns hand rank (lower is better):
/// - 1: Royal Flush
/// - 2-10: Straight Flush
/// - 11-166: Four of a Kind
/// - 167-322: Full House
/// - 323-1599: Flush
/// - 1600-1609: Straight
/// - 1610-2467: Three of a Kind
/// - 2468-3325: Two Pair
/// - 3326-6185: One Pair
/// - 6186-7462: High Card

const RANK_NAMES: [&str; 13] = [
    "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A",
];

const SUIT_NAMES: [&str; 4] = ["♠", "♥", "♦", "♣"];

/// Get rank from card (0-12: 2-A)
#[inline]
pub fn get_rank(card: u8) -> u8 {
    card % 13
}

/// Get suit from card (0-3: Spades, Hearts, Diamonds, Clubs)
#[inline]
pub fn get_suit(card: u8) -> u8 {
    card / 13
}

/// Convert card to string (e.g., "A♠")
pub fn card_to_string(card: u8) -> String {
    let rank = get_rank(card);
    let suit = get_suit(card);
    format!("{}{}", RANK_NAMES[rank as usize], SUIT_NAMES[suit as usize])
}

/// Fast 7-card hand evaluator
///
/// This is a simplified evaluator that's fast enough for River Hold'em.
/// For production, consider using a precomputed lookup table (133 MB).
///
/// Returns: Hand rank (1-7462, lower is better)
pub fn evaluate_7_cards(cards: &[u8; 7]) -> u32 {
    // Count ranks and suits
    let mut rank_counts = [0u8; 13];
    let mut suit_counts = [0u8; 4];

    for &card in cards {
        rank_counts[get_rank(card) as usize] += 1;
        suit_counts[get_suit(card) as usize] += 1;
    }

    // Check for flush
    let flush_suit = suit_counts.iter().position(|&count| count >= 5);

    // Check for straight
    let straight_high = find_straight(&rank_counts);

    // Royal Flush: A-K-Q-J-T suited
    if let Some(suit) = flush_suit {
        if let Some(high) = straight_high {
            if high == 12 {  // Ace-high straight
                let flush_cards: Vec<u8> = cards.iter()
                    .filter(|&&c| get_suit(c) == suit as u8)
                    .copied()
                    .collect();
                let mut flush_ranks = [0u8; 13];
                for &card in &flush_cards {
                    flush_ranks[get_rank(card) as usize] += 1;
                }
                if find_straight(&flush_ranks) == Some(12) {
                    return 1;  // Royal Flush
                }
            }
        }
    }

    // Straight Flush
    if let Some(suit) = flush_suit {
        let flush_cards: Vec<u8> = cards.iter()
            .filter(|&&c| get_suit(c) == suit as u8)
            .copied()
            .collect();
        let mut flush_ranks = [0u8; 13];
        for &card in &flush_cards {
            flush_ranks[get_rank(card) as usize] += 1;
        }
        if let Some(high) = find_straight(&flush_ranks) {
            return 2 + (12 - high) as u32;  // 2-10 for SF
        }
    }

    // Count pairs, trips, quads
    let mut quads = Vec::new();
    let mut trips = Vec::new();
    let mut pairs = Vec::new();
    let mut singles = Vec::new();

    for (rank, &count) in rank_counts.iter().enumerate() {
        match count {
            4 => quads.push(rank as u8),
            3 => trips.push(rank as u8),
            2 => pairs.push(rank as u8),
            1 => singles.push(rank as u8),
            _ => {}
        }
    }

    // Sort descending
    quads.sort_by(|a, b| b.cmp(a));
    trips.sort_by(|a, b| b.cmp(a));
    pairs.sort_by(|a, b| b.cmp(a));
    singles.sort_by(|a, b| b.cmp(a));

    // Four of a Kind
    if !quads.is_empty() {
        let quad_rank = quads[0];
        let kicker = if !singles.is_empty() {
            singles[0]
        } else if !pairs.is_empty() {
            pairs[0]
        } else {
            trips[0]
        };
        return 11 + (12 - quad_rank) as u32 * 13 + (12 - kicker) as u32;
    }

    // Full House
    if !trips.is_empty() && (!trips.len() < 2 && !pairs.is_empty() || trips.len() >= 2) {
        let trip_rank = trips[0];
        let pair_rank = if trips.len() >= 2 {
            trips[1]
        } else {
            pairs[0]
        };
        return 167 + (12 - trip_rank) as u32 * 13 + (12 - pair_rank) as u32;
    }

    // Flush
    if flush_suit.is_some() {
        // Get 5 highest flush cards
        let flush_ranks: Vec<u8> = cards.iter()
            .filter(|&&c| get_suit(c) == flush_suit.unwrap() as u8)
            .map(|&c| get_rank(c))
            .collect();
        let mut sorted_ranks: Vec<u8> = flush_ranks.iter().copied().collect();
        sorted_ranks.sort_by(|a, b| b.cmp(a));
        sorted_ranks.truncate(5);

        return 323 + rank_hand(&sorted_ranks);
    }

    // Straight
    if let Some(high) = straight_high {
        return 1600 + (12 - high) as u32;
    }

    // Three of a Kind
    if !trips.is_empty() {
        let trip_rank = trips[0];
        let mut kickers = singles.clone();
        kickers.extend(pairs);
        kickers.sort_by(|a, b| b.cmp(a));
        kickers.truncate(2);
        return 1610 + (12 - trip_rank) as u32 * 78
            + (12 - kickers.get(0).copied().unwrap_or(0)) as u32 * 13
            + (12 - kickers.get(1).copied().unwrap_or(0)) as u32;
    }

    // Two Pair
    if pairs.len() >= 2 {
        let pair1 = pairs[0];
        let pair2 = pairs[1];
        let kicker = if !singles.is_empty() {
            singles[0]
        } else if pairs.len() >= 3 {
            pairs[2]
        } else {
            trips[0]
        };
        return 2468 + (12 - pair1) as u32 * 169
            + (12 - pair2) as u32 * 13
            + (12 - kicker) as u32;
    }

    // One Pair
    if !pairs.is_empty() {
        let pair_rank = pairs[0];
        let mut kickers = singles.clone();
        kickers.extend(trips);
        kickers.sort_by(|a, b| b.cmp(a));
        kickers.truncate(3);
        return 3326 + (12 - pair_rank) as u32 * 220
            + rank_kickers(&kickers);
    }

    // High Card
    let mut all_ranks: Vec<u8> = singles.iter().copied().collect();
    all_ranks.sort_by(|a, b| b.cmp(a));
    all_ranks.truncate(5);
    6186 + rank_hand(&all_ranks)
}

/// Find highest straight in rank counts (returns high card rank or None)
fn find_straight(rank_counts: &[u8; 13]) -> Option<u8> {
    // Check A-2-3-4-5 (wheel)
    if rank_counts[12] > 0 && rank_counts[0] > 0 && rank_counts[1] > 0
        && rank_counts[2] > 0 && rank_counts[3] > 0 {
        return Some(3);  // 5-high straight
    }

    // Check regular straights (5-high to A-high)
    for high in (4..=12).rev() {
        if (0..5).all(|i| rank_counts[high - i] > 0) {
            return Some(high as u8);
        }
    }

    None
}

/// Rank a 5-card hand by high cards
fn rank_hand(ranks: &[u8]) -> u32 {
    ranks.iter().take(5).enumerate()
        .map(|(i, &r)| (12 - r) as u32 * 13u32.pow(4 - i as u32))
        .sum()
}

/// Rank kickers (3 cards)
fn rank_kickers(kickers: &[u8]) -> u32 {
    kickers.iter().take(3).enumerate()
        .map(|(i, &r)| (12 - r) as u32 * 13u32.pow(2 - i as u32))
        .sum()
}

/// Get hand rank category (0-9)
/// 0: High Card, 1: One Pair, ..., 9: Royal Flush
pub fn get_hand_category(rank: u32) -> u8 {
    match rank {
        1 => 9,           // Royal Flush
        2..=10 => 8,      // Straight Flush
        11..=166 => 7,    // Four of a Kind
        167..=322 => 6,   // Full House
        323..=1599 => 5,  // Flush
        1600..=1609 => 4, // Straight
        1610..=2467 => 3, // Three of a Kind
        2468..=3325 => 2, // Two Pair
        3326..=6185 => 1, // One Pair
        _ => 0,           // High Card
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_royal_flush() {
        // A♠ K♠ Q♠ J♠ T♠ 3♦ 2♥
        let cards = [12, 11, 10, 9, 8, 28, 14];  // Spades: 0-12
        let rank = evaluate_7_cards(&cards);
        assert_eq!(rank, 1, "Should be Royal Flush");
    }

    #[test]
    fn test_four_of_a_kind() {
        // A♠ A♥ A♦ A♣ K♠ Q♠ J♠
        let cards = [12, 25, 38, 51, 11, 10, 9];
        let rank = evaluate_7_cards(&cards);
        assert!(rank >= 11 && rank <= 166, "Should be Four of a Kind");
    }

    #[test]
    fn test_high_card() {
        // A♠ K♥ Q♦ J♣ 9♠ 7♥ 5♦
        let cards = [12, 24, 36, 48, 7, 19, 29];
        let rank = evaluate_7_cards(&cards);
        assert!(rank >= 6186, "Should be High Card");
    }
}
