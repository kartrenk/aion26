/// Full Heads-Up No-Limit Texas Hold'em game state implementation
///
/// This module provides a complete multi-street HUNL implementation
/// with configurable bet sizing (8 actions).

use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use crate::evaluator::{evaluate_7_cards, get_hand_category};

/// Betting streets
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Street {
    Preflop = 0,
    Flop = 1,
    Turn = 2,
    River = 3,
}

impl Street {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Street::Preflop,
            1 => Street::Flop,
            2 => Street::Turn,
            3 => Street::River,
            _ => Street::River,
        }
    }

}

/// Action types with bet sizing
/// 0 = Fold
/// 1 = Check/Call
/// 2 = Bet 0.5x pot
/// 3 = Bet 0.75x pot
/// 4 = Bet 1.0x pot
/// 5 = Bet 1.5x pot
/// 6 = Bet 2.0x pot
/// 7 = All-In
pub const ACTION_FOLD: u8 = 0;
pub const ACTION_CHECK_CALL: u8 = 1;
pub const ACTION_BET_HALF: u8 = 2;
pub const ACTION_BET_75: u8 = 3;
pub const ACTION_BET_POT: u8 = 4;
pub const ACTION_BET_150: u8 = 5;
pub const ACTION_BET_2X: u8 = 6;
pub const ACTION_ALLIN: u8 = 7;

/// Full HUNL game state
#[pyclass]
#[derive(Clone)]
pub struct RustFullHoldem {
    // Card state
    #[pyo3(get)]
    pub board: Vec<u8>,              // 0-5 community cards
    #[pyo3(get)]
    pub hands: Vec<Vec<u8>>,         // [2][2] - hole cards
    pub deck: Vec<u8>,               // Remaining deck for dealing

    // Street tracking
    #[pyo3(get)]
    pub street: u8,                  // 0-3 (Preflop, Flop, Turn, River)

    // Chip state
    #[pyo3(get)]
    pub pot: f64,
    #[pyo3(get)]
    pub stacks: Vec<f64>,            // [2] player stacks
    #[pyo3(get)]
    pub current_bet: f64,            // Current bet to call
    #[pyo3(get)]
    pub invested_street: Vec<f64>,   // [2] invested this street
    #[pyo3(get)]
    pub invested_total: Vec<f64>,    // [2] total invested in hand

    // Betting state
    #[pyo3(get)]
    pub actions_this_street: u8,     // Number of betting actions this street
    pub last_aggressor: i8,          // Last player to bet/raise (-1 if none)

    // Game state
    #[pyo3(get)]
    pub history: String,             // Action history (for info states)
    pub action_history: Vec<u8>,     // Numeric action history
    #[pyo3(get)]
    pub is_dealt: bool,              // Have hole cards been dealt?
    pub small_blind: f64,
    pub big_blind: f64,

    // Cached state
    current_player_cache: i8,
    is_terminal_cache: bool,
    needs_deal: bool,                // Need to deal community cards

    // Fixed flop for single-flop training mode
    pub fixed_flop: Option<Vec<u8>>, // If set, always deal this flop (3 cards)
}

#[pymethods]
impl RustFullHoldem {
    #[new]
    #[pyo3(signature = (
        stacks = vec![100.0, 100.0],
        small_blind = 0.5,
        big_blind = 1.0,
        fixed_flop = None,
    ))]
    pub fn new(
        stacks: Vec<f64>,
        small_blind: f64,
        big_blind: f64,
        fixed_flop: Option<Vec<u8>>,
    ) -> Self {
        let deck: Vec<u8> = (0..52).collect();

        RustFullHoldem {
            board: Vec::new(),
            hands: vec![Vec::new(), Vec::new()],
            deck,
            street: 0,  // Preflop
            pot: 0.0,
            stacks,
            current_bet: 0.0,
            invested_street: vec![0.0, 0.0],
            invested_total: vec![0.0, 0.0],
            actions_this_street: 0,
            last_aggressor: -1,
            history: String::new(),
            action_history: Vec::new(),
            is_dealt: false,
            small_blind,
            big_blind,
            current_player_cache: -1,  // Chance node
            is_terminal_cache: false,
            needs_deal: true,          // Need to deal hole cards
            fixed_flop,
        }
    }

    /// Get legal actions for current state
    pub fn legal_actions(&self) -> Vec<u8> {
        // Chance node - deal action
        if self.needs_deal {
            return vec![0];  // Deal action
        }

        if self.is_terminal_cache {
            return Vec::new();
        }

        let current_player = self.current_player();
        if current_player == -1 {
            return Vec::new();
        }

        let player_idx = current_player as usize;
        let to_call = self.current_bet - self.invested_street[player_idx];
        let stack = self.stacks[player_idx];

        let mut actions = Vec::new();

        // All-in situation: can only fold or call
        if to_call >= stack {
            actions.push(ACTION_FOLD);
            actions.push(ACTION_CHECK_CALL);
            return actions;
        }

        if to_call > 0.0 {
            // Facing a bet
            actions.push(ACTION_FOLD);
            actions.push(ACTION_CHECK_CALL);  // Call

            // Raises - must be at least min-raise
            let min_raise = self.current_bet * 2.0 - self.invested_street[player_idx];
            let pot_after_call = self.pot + to_call;

            // Add legal raise sizes
            for action in [ACTION_BET_HALF, ACTION_BET_75, ACTION_BET_POT, ACTION_BET_150, ACTION_BET_2X] {
                let raise_to = self.get_bet_amount(action, pot_after_call);
                if raise_to >= min_raise && raise_to < stack {
                    actions.push(action);
                }
            }

            // All-in is always legal
            actions.push(ACTION_ALLIN);
        } else {
            // No bet to call - can check or bet
            actions.push(ACTION_CHECK_CALL);  // Check

            // Bets - minimum is big blind
            let pot = self.pot;

            for action in [ACTION_BET_HALF, ACTION_BET_75, ACTION_BET_POT, ACTION_BET_150, ACTION_BET_2X] {
                let bet_size = self.get_bet_amount(action, pot);
                if bet_size >= self.big_blind && bet_size < stack {
                    actions.push(action);
                }
            }

            // All-in is always legal (if stack > 0)
            if stack > 0.0 {
                actions.push(ACTION_ALLIN);
            }
        }

        actions
    }

    /// Calculate bet amount for a given action
    fn get_bet_amount(&self, action: u8, pot_size: f64) -> f64 {
        match action {
            ACTION_BET_HALF => pot_size * 0.5,
            ACTION_BET_75 => pot_size * 0.75,
            ACTION_BET_POT => pot_size,
            ACTION_BET_150 => pot_size * 1.5,
            ACTION_BET_2X => pot_size * 2.0,
            _ => 0.0,
        }
    }

    /// Apply an action and return new game state
    pub fn apply_action(&self, action: u8) -> PyResult<Self> {
        let mut new_state = self.clone();

        // Handle chance nodes (dealing)
        if self.needs_deal {
            new_state.deal_next();
            return Ok(new_state);
        }

        let current_player = self.current_player();
        if current_player == -1 {
            return Ok(new_state);  // Terminal state
        }

        let player_idx = current_player as usize;
        let to_call = self.current_bet - self.invested_street[player_idx];
        let stack = self.stacks[player_idx];

        // Record action
        new_state.action_history.push(action);
        new_state.actions_this_street += 1;

        match action {
            ACTION_FOLD => {
                new_state.history.push('f');
                new_state.is_terminal_cache = true;
                new_state.current_player_cache = -1;
            }
            ACTION_CHECK_CALL => {
                if to_call > 0.0 {
                    // Call
                    let call_amount = to_call.min(stack);
                    new_state.stacks[player_idx] -= call_amount;
                    new_state.pot += call_amount;
                    new_state.invested_street[player_idx] += call_amount;
                    new_state.invested_total[player_idx] += call_amount;
                    new_state.history.push('c');

                    // Check if betting round is over
                    new_state.check_street_complete(current_player);
                } else {
                    // Check
                    new_state.history.push('k');

                    // Check if betting round is over
                    new_state.check_street_complete(current_player);
                }
            }
            ACTION_BET_HALF | ACTION_BET_75 | ACTION_BET_POT | ACTION_BET_150 | ACTION_BET_2X => {
                // Calculate pot after calling (if any)
                let pot_for_sizing = if to_call > 0.0 {
                    self.pot + to_call
                } else {
                    self.pot
                };

                let bet_size = self.get_bet_amount(action, pot_for_sizing);
                let total_bet = if to_call > 0.0 {
                    // This is a raise
                    let call_first = to_call.min(stack);
                    let raise_amount = bet_size.min(stack - call_first);
                    call_first + raise_amount
                } else {
                    // This is a bet
                    bet_size.min(stack)
                };

                new_state.stacks[player_idx] -= total_bet;
                new_state.pot += total_bet;
                new_state.invested_street[player_idx] += total_bet;
                new_state.invested_total[player_idx] += total_bet;
                new_state.current_bet = new_state.invested_street[player_idx];
                new_state.last_aggressor = current_player;

                let action_char = match action {
                    ACTION_BET_HALF => 'h',
                    ACTION_BET_75 => 's',  // 's' for small-ish
                    ACTION_BET_POT => 'b',
                    ACTION_BET_150 => 'm',  // 'm' for medium-plus
                    ACTION_BET_2X => 'd',   // 'd' for double
                    _ => 'b',
                };
                new_state.history.push(action_char);

                // Switch to opponent
                new_state.current_player_cache = 1 - current_player;
            }
            ACTION_ALLIN => {
                let all_in_amount = stack;

                new_state.stacks[player_idx] = 0.0;
                new_state.pot += all_in_amount;
                new_state.invested_street[player_idx] += all_in_amount;
                new_state.invested_total[player_idx] += all_in_amount;
                new_state.current_bet = new_state.invested_street[player_idx];
                new_state.last_aggressor = current_player;
                new_state.history.push('a');

                // Check if opponent can act
                let opp_idx = (1 - current_player) as usize;
                let opp_to_call = new_state.current_bet - new_state.invested_street[opp_idx];

                if new_state.stacks[opp_idx] > 0.0 && opp_to_call > 0.0 {
                    // Opponent can still act
                    new_state.current_player_cache = 1 - current_player;
                } else {
                    // Showdown (opponent is also all-in or called)
                    new_state.advance_to_showdown();
                }
            }
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid action: {}", action)
            )),
        }

        Ok(new_state)
    }

    /// Check if betting round is complete and advance street if needed
    fn check_street_complete(&mut self, acting_player: i8) {
        let bets_equal = (self.invested_street[0] - self.invested_street[1]).abs() < 0.001;

        // Check for check-check or bet-call
        let street_complete = if self.actions_this_street >= 2 && bets_equal {
            true
        } else if self.actions_this_street == 2 && self.history.ends_with("kk") {
            true
        } else {
            false
        };

        // Also check for preflop completion (BB checks after limp or calls raise)
        let preflop_complete = self.street == 0 && self.actions_this_street >= 2 && bets_equal;

        if street_complete || preflop_complete {
            // Check if someone is all-in
            if self.stacks[0] < 0.001 || self.stacks[1] < 0.001 {
                self.advance_to_showdown();
            } else if self.street == 3 {
                // River complete - showdown
                self.is_terminal_cache = true;
                self.current_player_cache = -1;
            } else {
                // Advance to next street
                self.advance_street();
            }
        } else {
            // Switch to opponent
            self.current_player_cache = 1 - acting_player;
        }
    }

    /// Advance to next street
    fn advance_street(&mut self) {
        self.street += 1;
        self.current_bet = 0.0;
        self.invested_street = vec![0.0, 0.0];
        self.actions_this_street = 0;
        self.last_aggressor = -1;
        self.history.push('/');

        // Need to deal community cards
        self.needs_deal = true;
        self.current_player_cache = -1;  // Chance node
    }

    /// Skip to showdown (when someone is all-in)
    fn advance_to_showdown(&mut self) {
        // Deal remaining cards if needed
        while self.board.len() < 5 {
            let card = self.deck.pop().unwrap();
            self.board.push(card);
        }

        self.street = 3;  // River
        self.is_terminal_cache = true;
        self.current_player_cache = -1;
    }

    /// Deal cards for current chance node
    fn deal_next(&mut self) {
        if !self.is_dealt {
            // Deal hole cards and post blinds
            self.deal_hole_cards();
            self.post_blinds();
            self.is_dealt = true;
            self.needs_deal = false;
            self.current_player_cache = 0;  // UTG acts first preflop (BTN/SB)
        } else {
            // Deal community cards for new street
            match Street::from_u8(self.street) {
                Street::Flop => {
                    // Use fixed flop if provided, otherwise deal randomly
                    if let Some(ref fixed_flop) = self.fixed_flop {
                        // Fixed flop mode: use provided cards
                        for &card in fixed_flop.iter().take(3) {
                            self.board.push(card);
                            // Remove from deck
                            if let Some(pos) = self.deck.iter().position(|&c| c == card) {
                                self.deck.remove(pos);
                            }
                        }
                    } else {
                        // Random flop: deal 3 cards from shuffled deck
                        for _ in 0..3 {
                            let card = self.deck.pop().unwrap();
                            self.board.push(card);
                        }
                    }
                }
                Street::Turn | Street::River => {
                    // Deal 1 card (always random)
                    let card = self.deck.pop().unwrap();
                    self.board.push(card);
                }
                _ => {}
            }

            self.needs_deal = false;
            self.current_player_cache = 1;  // BB (OOP) acts first postflop
        }
    }

    /// Deal hole cards
    fn deal_hole_cards(&mut self) {
        let mut rng = thread_rng();
        self.deck.shuffle(&mut rng);

        // Deal 2 cards to each player
        self.hands[0] = vec![self.deck.pop().unwrap(), self.deck.pop().unwrap()];
        self.hands[1] = vec![self.deck.pop().unwrap(), self.deck.pop().unwrap()];
    }

    /// Post blinds
    fn post_blinds(&mut self) {
        // Player 0 = BTN/SB, Player 1 = BB
        let sb_amount = self.small_blind.min(self.stacks[0]);
        let bb_amount = self.big_blind.min(self.stacks[1]);

        self.stacks[0] -= sb_amount;
        self.stacks[1] -= bb_amount;
        self.pot = sb_amount + bb_amount;

        self.invested_street[0] = sb_amount;
        self.invested_street[1] = bb_amount;
        self.invested_total[0] = sb_amount;
        self.invested_total[1] = bb_amount;

        self.current_bet = bb_amount;
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
        self.needs_deal
    }

    /// Get returns for terminal state
    pub fn returns(&self) -> Vec<f64> {
        if !self.is_terminal_cache {
            return vec![0.0, 0.0];
        }

        // Check if someone folded
        if self.history.contains('f') {
            // Find who folded by counting actions
            let folder = self.find_folder();
            let winner = 1 - folder;

            if winner == 0 {
                vec![self.pot - self.invested_total[0], -self.invested_total[1]]
            } else {
                vec![-self.invested_total[0], self.pot - self.invested_total[1]]
            }
        } else {
            // Showdown - need all 5 board cards
            if self.board.len() < 5 {
                // Shouldn't happen, but handle gracefully
                return vec![0.0, 0.0];
            }

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
                // Player 0 wins (lower rank = better)
                vec![self.pot - self.invested_total[0], -self.invested_total[1]]
            } else if rank1 < rank0 {
                // Player 1 wins
                vec![-self.invested_total[0], self.pot - self.invested_total[1]]
            } else {
                // Tie - split pot
                let split = self.pot / 2.0;
                vec![split - self.invested_total[0], split - self.invested_total[1]]
            }
        }
    }

    /// Find which player folded
    fn find_folder(&self) -> i8 {
        // Parse history to find folder
        // Player 0 acts on odd positions preflop, even positions postflop
        let mut player = if self.history.starts_with('/') { 1 } else { 0 };

        for ch in self.history.chars() {
            if ch == 'f' {
                return player;
            } else if ch == '/' {
                player = 1;  // OOP acts first postflop
            } else {
                player = 1 - player;
            }
        }

        0  // Default (shouldn't reach here)
    }

    /// Get hand strength category (for encoder) - only valid on river
    pub fn get_hand_strength(&self, player: u8) -> u8 {
        if self.board.len() < 5 {
            return 0;  // No hand category until river
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

    /// Get information state string for player
    pub fn information_state_string(&self, player: u8) -> String {
        if !self.is_dealt {
            return "chance".to_string();
        }

        let hand = &self.hands[player as usize];

        // Canonicalize hole cards by sorting
        let mut sorted_hand = hand.clone();
        sorted_hand.sort_by_key(|&card| {
            let rank = card % 13;
            let suit = card / 13;
            (12 - rank, suit)
        });

        let mut state = format!("h{}{}", sorted_hand[0], sorted_hand[1]);

        // Add visible board cards
        if !self.board.is_empty() {
            state.push_str(&format!("b{}",
                self.board.iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join("")
            ));
        }

        // Add history
        if !self.history.is_empty() {
            state.push_str(&format!(":{}", self.history));
        }

        state
    }

    /// Get action history as Python list
    pub fn get_action_history(&self) -> Vec<u8> {
        self.action_history.clone()
    }
}
