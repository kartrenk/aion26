/// Fast River Hold'em game state implementation
///
/// This module provides a Rust implementation of Texas Hold'em River
/// for 50-100x speedup over Python.

use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use crate::evaluator::{evaluate_7_cards, get_hand_category};

/// River Hold'em game state
#[pyclass]
#[derive(Clone)]
pub struct RustRiverHoldem {
    // Card state
    #[pyo3(get)]
    pub board: Vec<u8>,
    #[pyo3(get)]
    pub hands: Vec<Vec<u8>>,
    pub deck: Vec<u8>,
    pub fixed_board: Option<Vec<u8>>,  // If Some, use this board instead of random

    // Chip state
    #[pyo3(get)]
    pub pot: f64,
    #[pyo3(get)]
    pub stacks: Vec<f64>,
    #[pyo3(get)]
    pub current_bet: f64,
    #[pyo3(get)]
    pub player_0_invested: f64,
    #[pyo3(get)]
    pub player_1_invested: f64,

    // Game state
    #[pyo3(get)]
    pub history: String,
    #[pyo3(get)]
    pub is_dealt: bool,
    pub current_player_cache: i8,
    pub is_terminal_cache: bool,
}

#[pymethods]
impl RustRiverHoldem {
    #[new]
    #[pyo3(signature = (
        stacks = vec![100.0, 100.0],
        pot = 2.0,
        current_bet = 0.0,
        player_0_invested = 1.0,
        player_1_invested = 1.0,
        fixed_board = None,
    ))]
    pub fn new(
        stacks: Vec<f64>,
        pot: f64,
        current_bet: f64,
        player_0_invested: f64,
        player_1_invested: f64,
        fixed_board: Option<Vec<u8>>,
    ) -> Self {
        let deck: Vec<u8> = (0..52).collect();

        RustRiverHoldem {
            board: Vec::new(),
            hands: vec![Vec::new(), Vec::new()],
            deck,
            fixed_board,
            pot,
            stacks,
            current_bet,
            player_0_invested,
            player_1_invested,
            history: String::new(),
            is_dealt: false,
            current_player_cache: -1,  // Chance node
            is_terminal_cache: false,
        }
    }

    /// Get legal actions for current state
    pub fn legal_actions(&self) -> Vec<u8> {
        if !self.is_dealt {
            return vec![0];  // Deal action
        }

        if self.is_terminal_cache {
            return Vec::new();
        }

        let current_player = self.current_player();
        if current_player == -1 {
            return Vec::new();  // Chance or terminal
        }

        let invested = if current_player == 0 {
            self.player_0_invested
        } else {
            self.player_1_invested
        };

        let to_call = self.current_bet - invested;
        let stack = self.stacks[current_player as usize];

        if to_call >= stack {
            // All-in situation: can only fold or call all-in
            vec![0, 1]  // Fold, Call
        } else if to_call > 0.0 {
            // Facing a bet: fold, call, raise pot, all-in
            vec![0, 1, 2, 3]
        } else {
            // No bet: check, bet pot, all-in
            vec![1, 2, 3]  // Check, Bet Pot, All-In
        }
    }

    /// Apply an action and return new game state
    pub fn apply_action(&self, action: u8) -> PyResult<Self> {
        let mut new_state = self.clone();

        if !self.is_dealt {
            // Deal cards
            new_state.deal_cards();
            new_state.is_dealt = true;
            new_state.current_player_cache = 0;  // Player 0 acts first
            return Ok(new_state);
        }

        let current_player = self.current_player();
        if current_player == -1 {
            return Ok(new_state);  // Terminal or chance
        }

        let player_idx = current_player as usize;
        let invested = if current_player == 0 {
            self.player_0_invested
        } else {
            self.player_1_invested
        };

        let to_call = self.current_bet - invested;

        match action {
            0 => {
                // Fold
                new_state.history.push('f');
                new_state.is_terminal_cache = true;
                new_state.current_player_cache = -1;
            }
            1 => {
                // Check/Call
                if to_call > 0.0 {
                    // Call
                    let call_amount = to_call.min(new_state.stacks[player_idx]);
                    new_state.stacks[player_idx] -= call_amount;
                    new_state.pot += call_amount;

                    if current_player == 0 {
                        new_state.player_0_invested += call_amount;
                    } else {
                        new_state.player_1_invested += call_amount;
                    }

                    new_state.history.push('c');
                    new_state.is_terminal_cache = true;  // Showdown
                    new_state.current_player_cache = -1;
                } else {
                    // Check
                    new_state.history.push('k');

                    if self.history.contains('k') {
                        // Both checked - showdown
                        new_state.is_terminal_cache = true;
                        new_state.current_player_cache = -1;
                    } else {
                        // Switch players
                        new_state.current_player_cache = 1 - current_player;
                    }
                }
            }
            2 => {
                // Bet Pot
                let bet_size = new_state.pot;
                let actual_bet = bet_size.min(new_state.stacks[player_idx]);

                new_state.stacks[player_idx] -= actual_bet;
                new_state.pot += actual_bet;
                new_state.current_bet = invested + actual_bet;

                if current_player == 0 {
                    new_state.player_0_invested += actual_bet;
                } else {
                    new_state.player_1_invested += actual_bet;
                }

                new_state.history.push('b');
                new_state.current_player_cache = 1 - current_player;
            }
            3 => {
                // All-In
                let all_in_amount = new_state.stacks[player_idx];

                new_state.stacks[player_idx] = 0.0;
                new_state.pot += all_in_amount;
                new_state.current_bet = invested + all_in_amount;

                if current_player == 0 {
                    new_state.player_0_invested += all_in_amount;
                } else {
                    new_state.player_1_invested += all_in_amount;
                }

                new_state.history.push('a');
                new_state.current_player_cache = 1 - current_player;
            }
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid action: {}", action)
            )),
        }

        Ok(new_state)
    }

    /// Check if game is terminal
    pub fn is_terminal(&self) -> bool {
        self.is_terminal_cache
    }

    /// Get current player (-1 for chance/terminal, 0/1 for players)
    pub fn current_player(&self) -> i8 {
        self.current_player_cache
    }

    /// Check if this is a chance node
    pub fn is_chance_node(&self) -> bool {
        !self.is_dealt
    }

    /// Get returns for terminal state
    pub fn returns(&self) -> Vec<f64> {
        if !self.is_terminal_cache {
            return vec![0.0, 0.0];
        }

        // Check if someone folded
        if self.history.ends_with('f') {
            let last_player = if self.history.len() % 2 == 1 {
                0
            } else {
                1
            };

            let winner = 1 - last_player;
            let winnings = self.pot;

            if winner == 0 {
                vec![winnings - self.player_0_invested, -self.player_1_invested]
            } else {
                vec![-self.player_0_invested, winnings - self.player_1_invested]
            }
        } else {
            // Showdown
            let hand0 = [
                self.hands[0][0], self.hands[0][1],
                self.board[0], self.board[1], self.board[2],
                self.board[3], self.board[4],
            ];
            let hand1 = [
                self.hands[1][0], self.hands[1][1],
                self.board[0], self.board[1], self.board[2],
                self.board[3], self.board[4],
            ];

            let rank0 = evaluate_7_cards(&hand0);
            let rank1 = evaluate_7_cards(&hand1);

            if rank0 < rank1 {
                // Player 0 wins
                vec![self.pot - self.player_0_invested, -self.player_1_invested]
            } else if rank1 < rank0 {
                // Player 1 wins
                vec![-self.player_0_invested, self.pot - self.player_1_invested]
            } else {
                // Tie - split pot
                let split = self.pot / 2.0;
                vec![split - self.player_0_invested, split - self.player_1_invested]
            }
        }
    }

    /// Get information state string for player
    pub fn information_state_string(&self, player: u8) -> String {
        if !self.is_dealt {
            return "chance".to_string();
        }

        let hand = &self.hands[player as usize];

        // CRITICAL: Canonicalize hole cards by sorting
        // This ensures [Ah, 2c] and [2c, Ah] map to the same info state
        // Cards are 0-51: rank = card % 13, suit = card / 13
        let mut sorted_hand = hand.clone();
        sorted_hand.sort_by_key(|&card| {
            let rank = card % 13;
            let suit = card / 13;
            // Sort by rank (descending), then suit
            // This gives consistent ordering: Ah2c not 2cAh
            (12 - rank, suit)  // 12 - rank for descending (Ace highest)
        });

        let mut state = format!("h{}{}", sorted_hand[0], sorted_hand[1]);

        // Add board (board order doesn't matter for hand value, but keep for consistency)
        state.push_str(&format!("b{}{}{}{}{}",
            self.board[0], self.board[1], self.board[2],
            self.board[3], self.board[4]));

        // Add history
        if !self.history.is_empty() {
            state.push_str(&format!(":{}", self.history));
        }

        state
    }

    /// Deal random cards
    fn deal_cards(&mut self) {
        let mut rng = thread_rng();

        if let Some(ref fixed_board) = self.fixed_board {
            // Fixed board mode: use provided board, deal random hole cards from remaining 47 cards
            // Board is already fixed (e.g., [As, Ks, Qs, Js, 2h] = [12, 25, 38, 51, 1])
            self.board = fixed_board.clone();

            // Create deck without the fixed board cards
            let mut available_cards: Vec<u8> = (0..52)
                .filter(|c| !fixed_board.contains(c))
                .collect();

            available_cards.shuffle(&mut rng);

            // Deal 2 cards to each player from remaining 47 cards
            self.hands[0] = vec![available_cards[0], available_cards[1]];
            self.hands[1] = vec![available_cards[2], available_cards[3]];
        } else {
            // Random board mode: shuffle entire deck
            let mut shuffled = self.deck.clone();
            shuffled.shuffle(&mut rng);

            self.hands[0] = vec![shuffled[0], shuffled[1]];
            self.hands[1] = vec![shuffled[2], shuffled[3]];
            self.board = vec![shuffled[4], shuffled[5], shuffled[6], shuffled[7], shuffled[8]];
        }
    }

    /// Get hand strength category (for encoder)
    pub fn get_hand_strength(&self, player: u8) -> u8 {
        if !self.is_dealt {
            return 0;
        }

        let hand = &self.hands[player as usize];
        let cards = [
            hand[0], hand[1],
            self.board[0], self.board[1], self.board[2],
            self.board[3], self.board[4],
        ];

        let rank = evaluate_7_cards(&cards);
        get_hand_category(rank)
    }
}
