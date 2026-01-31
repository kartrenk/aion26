/// Parallel Deep CFR Trainer for Full HUNL
///
/// Multi-street traversal with 8 actions and 220-dim state encoding.

use pyo3::prelude::*;
use numpy::{PyArray2, PyArrayMethods};
use rayon::prelude::*;
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::path::PathBuf;
use rand::Rng;

use crate::game_full::RustFullHoldem;
use crate::evaluator::{evaluate_7_cards, get_hand_category};
use crate::io_full::{TrajectoryWriterFull, STATE_DIM, TARGET_DIM};

// ============================================================================
// Configuration
// ============================================================================

const MIN_BATCH_SIZE: usize = 2048;

/// Number of board texture buckets for flop abstraction
const NUM_TEXTURE_BUCKETS: usize = 200;

// ============================================================================
// Board Texture Bucketing (Inline)
// ============================================================================

/// Compute board texture bucket ID from flop cards
/// This is a fast inline computation that doesn't require precomputed K-Means
/// Features: suit pattern, connectedness, high card strength, paired status
fn compute_board_texture_bucket(board: &[u8]) -> usize {
    if board.len() < 3 {
        return 0;
    }

    let mut ranks: Vec<u8> = board.iter().take(3).map(|&c| c % 13).collect();
    let suits: Vec<u8> = board.iter().take(3).map(|&c| c / 13).collect();

    ranks.sort_by(|a, b| b.cmp(a)); // Descending

    // Feature 1: Suit pattern (0-2)
    let unique_suits: std::collections::HashSet<u8> = suits.iter().cloned().collect();
    let suit_feature = match unique_suits.len() {
        1 => 2,  // Monotone
        2 => 1,  // Two-tone
        _ => 0,  // Rainbow
    };

    // Feature 2: Connectedness (0-9)
    let gap1 = (ranks[0] as i32 - ranks[1] as i32).abs() as usize;
    let gap2 = (ranks[1] as i32 - ranks[2] as i32).abs() as usize;
    let connected_feature = (10 - (gap1 + gap2).min(10)).min(9);

    // Feature 3: High card (0-3)
    let high_feature = (ranks[0] / 4).min(3) as usize;

    // Feature 4: Paired (0-1)
    let paired_feature = if ranks[0] == ranks[1] || ranks[1] == ranks[2] { 1 } else { 0 };

    // Combine features into bucket ID
    // suit(3) × connected(10) × high(4) × paired(2) = 240 combinations
    // We'll mod to NUM_TEXTURE_BUCKETS
    let bucket = suit_feature * 80 + connected_feature * 8 + high_feature * 2 + paired_feature;
    bucket % NUM_TEXTURE_BUCKETS
}

// ============================================================================
// Shared State
// ============================================================================

/// Thread-safe query buffer for 220-dim states
struct SharedQueryBuffer {
    states: Mutex<Vec<f32>>,
    query_ids: Mutex<Vec<usize>>,
    count: AtomicUsize,
    next_query_id: AtomicUsize,
    max_queries: usize,
}

impl SharedQueryBuffer {
    fn new(max_queries: usize) -> Self {
        Self {
            states: Mutex::new(vec![0.0; max_queries * STATE_DIM]),
            query_ids: Mutex::new(Vec::with_capacity(max_queries)),
            count: AtomicUsize::new(0),
            next_query_id: AtomicUsize::new(0),
            max_queries,
        }
    }

    fn add_query(&self, state_encoding: &[f32]) -> Option<usize> {
        let count = self.count.load(Ordering::Relaxed);
        if count >= self.max_queries {
            return None;
        }

        let slot = self.count.fetch_add(1, Ordering::SeqCst);
        if slot >= self.max_queries {
            self.count.fetch_sub(1, Ordering::SeqCst);
            return None;
        }

        let query_id = self.next_query_id.fetch_add(1, Ordering::SeqCst);

        {
            let mut states = self.states.lock();
            let offset = slot * STATE_DIM;
            states[offset..offset + STATE_DIM].copy_from_slice(state_encoding);
        }
        {
            let mut query_ids = self.query_ids.lock();
            while query_ids.len() <= slot {
                query_ids.push(0);
            }
            query_ids[slot] = query_id;
        }

        Some(query_id)
    }

    fn count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    fn clear(&self) {
        self.count.store(0, Ordering::SeqCst);
        self.query_ids.lock().clear();
    }

    fn get_slice(&self) -> Vec<f32> {
        let count = self.count.load(Ordering::Relaxed);
        let states = self.states.lock();
        states[0..count * STATE_DIM].to_vec()
    }

    fn get_query_ids(&self) -> Vec<usize> {
        let count = self.count.load(Ordering::Relaxed);
        let query_ids = self.query_ids.lock();
        query_ids[0..count].to_vec()
    }
}

/// Thread-safe prediction cache
struct SharedPredictionCache {
    cache: Mutex<std::collections::HashMap<usize, Vec<f32>>>,
}

impl SharedPredictionCache {
    fn new() -> Self {
        Self {
            cache: Mutex::new(std::collections::HashMap::new()),
        }
    }

    fn get(&self, query_id: usize) -> Option<Vec<f32>> {
        self.cache.lock().get(&query_id).cloned()
    }

    fn insert(&self, query_id: usize, strategy: Vec<f32>) {
        self.cache.lock().insert(query_id, strategy);
    }

    fn clear(&self) {
        self.cache.lock().clear();
    }
}

/// Thread-safe sample writer
struct SharedWriter {
    writer: Mutex<Option<TrajectoryWriterFull>>,
    samples_written: AtomicUsize,
}

impl SharedWriter {
    fn new() -> Self {
        Self {
            writer: Mutex::new(None),
            samples_written: AtomicUsize::new(0),
        }
    }

    fn set_writer(&self, w: TrajectoryWriterFull) {
        *self.writer.lock() = Some(w);
    }

    fn write_sample(&self, state: &[f32], target: &[f32]) -> Result<(), String> {
        let mut guard = self.writer.lock();
        if let Some(ref mut writer) = *guard {
            writer.append(state.try_into().unwrap(), target.try_into().unwrap())
                .map_err(|e| e.to_string())?;
            self.samples_written.fetch_add(1, Ordering::Relaxed);
            Ok(())
        } else {
            Err("No writer".to_string())
        }
    }

    fn flush(&self) -> Result<usize, String> {
        let mut guard = self.writer.lock();
        if let Some(ref mut writer) = *guard {
            writer.flush().map_err(|e| e.to_string())?;
            Ok(writer.len())
        } else {
            Ok(0)
        }
    }

    fn samples(&self) -> usize {
        self.samples_written.load(Ordering::Relaxed)
    }

    fn close(&self) -> Result<usize, String> {
        let mut guard = self.writer.lock();
        if let Some(mut writer) = guard.take() {
            writer.flush().map_err(|e| e.to_string())?;
            Ok(writer.len())
        } else {
            Ok(0)
        }
    }
}

// ============================================================================
// Traversal Context
// ============================================================================

#[derive(Clone)]
struct TraversalFrame {
    game_state: RustFullHoldem,
    update_player: u8,
    reach_prob_0: f32,
    reach_prob_1: f32,
    query_id: Option<usize>,
    strategy: Option<Vec<f32>>,
    action_values: Vec<f32>,
    current_action: usize,
    is_update_player: bool,
    chance_processed: bool,
}

struct Worker {
    stack: Vec<TraversalFrame>,
    rng: rand::rngs::StdRng,
    small_blind: f64,
    big_blind: f64,
    starting_stack: f64,
    fixed_flop: Option<Vec<u8>>,  // If set, always use this flop
    completed: usize,
    target: usize,
    started: usize,
}

impl Worker {
    fn new(small_blind: f64, big_blind: f64, starting_stack: f64, fixed_flop: Option<Vec<u8>>) -> Self {
        Self {
            stack: Vec::with_capacity(64),  // Deeper stack for multi-street
            rng: rand::SeedableRng::from_entropy(),
            small_blind,
            big_blind,
            starting_stack,
            fixed_flop,
            completed: 0,
            target: 0,
            started: 0,
        }
    }

    fn set_target(&mut self, target: usize) {
        self.target = target;
        self.completed = 0;
        self.started = 0;
        self.stack.clear();
    }

    fn is_done(&self) -> bool {
        self.completed >= self.target
    }

    fn create_initial_game(&self) -> RustFullHoldem {
        RustFullHoldem::new(
            vec![self.starting_stack, self.starting_stack],
            self.small_blind,
            self.big_blind,
            self.fixed_flop.clone(),
        )
    }

    fn start_traversal(&mut self) {
        if self.started >= self.target {
            return;
        }
        self.started += 1;
        let update_player = (self.started % 2) as u8;

        self.stack.push(TraversalFrame {
            game_state: self.create_initial_game(),
            update_player,
            reach_prob_0: 1.0,
            reach_prob_1: 1.0,
            query_id: None,
            strategy: None,
            action_values: Vec::new(),
            current_action: 0,
            is_update_player: false,
            chance_processed: false,
        });
    }

    /// Process the traversal stack. Returns true if made progress, false if blocked.
    fn step(
        &mut self,
        query_buffer: &SharedQueryBuffer,
        prediction_cache: &SharedPredictionCache,
        writer: &SharedWriter,
    ) -> bool {
        // Start new traversal if stack empty
        if self.stack.is_empty() {
            if self.started < self.target {
                self.start_traversal();
                return true;
            }
            return false;
        }

        let frame = self.stack.last_mut().unwrap();

        // Handle terminal
        if frame.game_state.is_terminal() {
            let returns = frame.game_state.returns();
            let value = returns[frame.update_player as usize] as f32;
            return self.pop_with_value(value, writer);
        }

        // Handle chance node (deal)
        if frame.game_state.is_chance_node() {
            if frame.chance_processed {
                // Child already processed and returned, propagate value
                let value = frame.action_values.first().copied().unwrap_or(0.0);
                return self.pop_with_value(value, writer);
            }

            let next_state = frame.game_state.apply_action(0).expect("Deal failed");
            let update_player = frame.update_player;
            let reach_0 = frame.reach_prob_0;
            let reach_1 = frame.reach_prob_1;

            // Mark that we've spawned the child
            self.stack.last_mut().unwrap().chance_processed = true;

            self.stack.push(TraversalFrame {
                game_state: next_state,
                update_player,
                reach_prob_0: reach_0,
                reach_prob_1: reach_1,
                query_id: None,
                strategy: None,
                action_values: Vec::new(),
                current_action: 0,
                is_update_player: false,
                chance_processed: false,
            });
            return true;
        }

        // Need strategy from neural network
        if frame.strategy.is_none() {
            // Check if we have a pending query
            if let Some(qid) = frame.query_id {
                if let Some(strat) = prediction_cache.get(qid) {
                    let current_player = frame.game_state.current_player();
                    frame.is_update_player = current_player as u8 == frame.update_player;
                    let num_actions = frame.game_state.legal_actions().len();
                    frame.action_values = vec![0.0; num_actions];
                    frame.strategy = Some(strat);
                    return true;
                } else {
                    // Still waiting for inference
                    return false;
                }
            } else {
                // Submit query
                let state_encoding = encode_state(&frame.game_state, frame.update_player);
                if let Some(qid) = query_buffer.add_query(&state_encoding) {
                    frame.query_id = Some(qid);
                }
                return false;
            }
        }

        // Have strategy, explore actions
        let strategy = frame.strategy.as_ref().unwrap().clone();
        let legal_actions = frame.game_state.legal_actions();
        let num_actions = legal_actions.len();
        let is_update = frame.is_update_player;
        let current_action = frame.current_action;
        let update_player = frame.update_player;
        let current_player = frame.game_state.current_player();
        let reach_0 = frame.reach_prob_0;
        let reach_1 = frame.reach_prob_1;

        if is_update {
            // Update player: explore ALL actions
            if current_action < num_actions {
                let action = legal_actions[current_action];
                let frame = self.stack.last_mut().unwrap();
                let next_state = frame.game_state.apply_action(action).expect("Action failed");

                let prob = strategy.get(current_action).copied().unwrap_or(1.0 / num_actions as f32);
                let new_reach_0 = if current_player == 0 { reach_0 * prob } else { reach_0 };
                let new_reach_1 = if current_player == 1 { reach_1 * prob } else { reach_1 };

                self.stack.push(TraversalFrame {
                    game_state: next_state,
                    update_player,
                    reach_prob_0: new_reach_0,
                    reach_prob_1: new_reach_1,
                    query_id: None,
                    strategy: None,
                    action_values: Vec::new(),
                    current_action: 0,
                    is_update_player: false,
                    chance_processed: false,
                });
                return true;
            } else {
                // All actions explored, compute regrets and pop
                let frame = self.stack.last().unwrap();
                let ev: f32 = strategy.iter()
                    .zip(frame.action_values.iter())
                    .map(|(p, v)| p * v)
                    .sum();

                // Write regret sample (8 actions)
                let mut regrets = vec![0.0f32; TARGET_DIM];
                for (i, &av) in frame.action_values.iter().enumerate() {
                    if i < TARGET_DIM {
                        regrets[i] = av - ev;
                    }
                }

                // Scale by opponent reach
                let opp_reach = if update_player == 0 { reach_1 } else { reach_0 };
                for r in regrets.iter_mut() {
                    *r *= opp_reach;
                }

                let state = encode_state(&frame.game_state, update_player);
                let _ = writer.write_sample(&state, &regrets);

                return self.pop_with_value(ev, writer);
            }
        } else {
            // Opponent: sample ONE action
            if current_action < num_actions {
                let sampled_idx = self.sample_action(&strategy, num_actions);
                let frame = self.stack.last_mut().unwrap();

                let action = legal_actions[sampled_idx];
                let next_state = frame.game_state.apply_action(action).expect("Action failed");

                let prob = strategy.get(sampled_idx).copied().unwrap_or(1.0 / num_actions as f32);
                let new_reach_0 = if current_player == 0 { reach_0 * prob } else { reach_0 };
                let new_reach_1 = if current_player == 1 { reach_1 * prob } else { reach_1 };

                // Mark that we've "explored"
                frame.current_action = num_actions;

                self.stack.push(TraversalFrame {
                    game_state: next_state,
                    update_player,
                    reach_prob_0: new_reach_0,
                    reach_prob_1: new_reach_1,
                    query_id: None,
                    strategy: None,
                    action_values: Vec::new(),
                    current_action: 0,
                    is_update_player: false,
                    chance_processed: false,
                });
                return true;
            } else {
                // Opponent's child returned, propagate value
                let frame = self.stack.last().unwrap();
                let value = frame.action_values.first().copied().unwrap_or(0.0);
                return self.pop_with_value(value, writer);
            }
        }
    }

    fn pop_with_value(&mut self, value: f32, _writer: &SharedWriter) -> bool {
        self.stack.pop();

        if self.stack.is_empty() {
            // Traversal complete!
            self.completed += 1;
            return true;
        }

        // Propagate value to parent
        let parent = self.stack.last_mut().unwrap();
        if parent.is_update_player {
            // Update player: store value for this action
            if parent.current_action < parent.action_values.len() {
                parent.action_values[parent.current_action] = value;
            }
            parent.current_action += 1;
        } else {
            // Opponent: store value
            if parent.action_values.is_empty() {
                parent.action_values.push(value);
            } else {
                parent.action_values[0] = value;
            }
        }

        true
    }

    fn sample_action(&mut self, strategy: &[f32], num_actions: usize) -> usize {
        let r: f32 = self.rng.gen();
        let mut cumsum = 0.0;
        for (i, &p) in strategy.iter().take(num_actions).enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }
        num_actions.saturating_sub(1)
    }
}

// ============================================================================
// State Encoder (220 dims)
// ============================================================================

/// Canonicalize suits for isomorphic state representation.
/// Maps suits to canonical form (0,1,2,3) based on order of first appearance.
/// This is a LOSSLESS compression - Ah Kh 2s becomes equivalent to Ad Kd 2c.
/// Reduces the effective state space by ~12.5x for flops.
fn canonicalize_suits(cards: &[u8]) -> [i8; 4] {
    let mut suit_map: [i8; 4] = [-1, -1, -1, -1];
    let mut next_canonical_suit: i8 = 0;

    for &card in cards {
        let suit = (card / 13) as usize;
        if suit < 4 && suit_map[suit] == -1 {
            suit_map[suit] = next_canonical_suit;
            next_canonical_suit += 1;
        }
    }

    // Fill remaining unmapped suits
    for i in 0..4 {
        if suit_map[i] == -1 {
            suit_map[i] = next_canonical_suit;
            next_canonical_suit += 1;
        }
    }

    suit_map
}

fn encode_state(state: &RustFullHoldem, player: u8) -> Vec<f32> {
    let mut features = vec![0.0; STATE_DIM];

    if !state.is_dealt {
        return features;
    }

    let hand = &state.hands[player as usize];
    let board = &state.board;
    let street = state.street;

    // SUIT ISOMORPHISM: Build canonical suit mapping from all visible cards
    // Order: player's hole cards first, then board cards
    // This ensures Ah Kh on As Ks 2d is same as Ad Kd on Ah Kh 2c
    let all_cards: Vec<u8> = hand.iter().chain(board.iter()).cloned().collect();
    let suit_map = canonicalize_suits(&all_cards);

    // 1. Hand rank category (10 dims) - only on river
    // NOTE: Hand evaluation uses actual suits (flush detection) - this is correct
    // because we only canonicalize the ENCODING, not the game logic
    if board.len() >= 5 {
        let seven_cards: Vec<u8> = hand.iter().chain(board.iter().take(5)).cloned().collect();
        if seven_cards.len() == 7 {
            let rank = evaluate_7_cards(&seven_cards.try_into().unwrap());
            let category = get_hand_category(rank) as usize;
            if category < 10 {
                features[category] = 1.0;
            }
        }
    }

    // 2. Hole cards (34 dims): indices 10-43
    // Use CANONICAL suits for encoding
    for (i, &card) in hand.iter().enumerate().take(2) {
        let offset = 10 + i * 17;
        let rank = (card % 13) as usize;
        let actual_suit = (card / 13) as usize;
        let canonical_suit = suit_map[actual_suit] as usize;
        if rank < 13 { features[offset + rank] = 1.0; }
        if canonical_suit < 4 { features[offset + 13 + canonical_suit] = 1.0; }
    }

    // 3. Board cards (85 dims): indices 44-128
    // Use CANONICAL suits for encoding
    for (i, &card) in board.iter().enumerate().take(5) {
        let offset = 44 + i * 17;
        let rank = (card % 13) as usize;
        let actual_suit = (card / 13) as usize;
        let canonical_suit = suit_map[actual_suit] as usize;
        if rank < 13 { features[offset + rank] = 1.0; }
        if canonical_suit < 4 { features[offset + 13 + canonical_suit] = 1.0; }
    }

    // 4. Street one-hot (4 dims): indices 129-132
    let street_idx = (street as usize).min(3);
    features[129 + street_idx] = 1.0;

    // 5. Action history (64 dims): indices 133-196
    // Last 8 actions, each encoded as 8-dim one-hot
    let action_hist = state.get_action_history();
    let hist_len = action_hist.len();
    let start = if hist_len > 8 { hist_len - 8 } else { 0 };

    for (slot, &action) in action_hist[start..].iter().enumerate() {
        if slot < 8 {
            let offset = 133 + slot * 8;
            let action_idx = (action as usize).min(7);
            features[offset + action_idx] = 1.0;
        }
    }

    // 6. Betting context (23 dims): indices 197-219
    let ctx_offset = 197;

    // CRITICAL FIX: Use dynamic normalization based on total money in play
    // This ensures features are scale-invariant and consistent with Python inference
    let total_money = state.stacks[0] + state.stacks[1] + state.pot;
    let normalizer = if total_money > 0.0 { total_money as f32 } else { 1.0 };

    // Pot (normalized by total money)
    features[ctx_offset] = state.pot as f32 / normalizer;

    // Player stacks (2 dims)
    features[ctx_offset + 1] = state.stacks[0] as f32 / normalizer;
    features[ctx_offset + 2] = state.stacks[1] as f32 / normalizer;

    // Current bet (1 dim)
    features[ctx_offset + 3] = state.current_bet as f32 / normalizer;

    // Invested this street (2 dims)
    features[ctx_offset + 4] = state.invested_street[0] as f32 / normalizer;
    features[ctx_offset + 5] = state.invested_street[1] as f32 / normalizer;

    // Invested total (2 dims)
    features[ctx_offset + 6] = state.invested_total[0] as f32 / normalizer;
    features[ctx_offset + 7] = state.invested_total[1] as f32 / normalizer;

    // Pot odds
    let invested = state.invested_street[player as usize];
    let to_call = (state.current_bet - invested).max(0.0);
    let pot_after = state.pot + to_call;
    let pot_odds = if pot_after > 0.0 { to_call / pot_after } else { 0.0 };
    features[ctx_offset + 8] = pot_odds as f32;

    // Stack-to-pot ratio
    let my_stack = state.stacks[player as usize];
    let spr = if state.pot > 0.0 { my_stack / state.pot } else { 10.0 };
    features[ctx_offset + 9] = (spr as f32 / 10.0).min(1.0);

    // Position indicator (1 dim)
    features[ctx_offset + 10] = if player == 0 { 1.0 } else { 0.0 };

    // Actions this street
    features[ctx_offset + 11] = (state.actions_this_street as f32 / 10.0).min(1.0);

    // FLOP ABSTRACTION: Board texture bucket (normalized 0-1)
    // This provides the network with a compressed representation of board texture
    // that groups strategically similar flops together
    if board.len() >= 3 {
        let texture_bucket = compute_board_texture_bucket(board);
        features[ctx_offset + 12] = texture_bucket as f32 / NUM_TEXTURE_BUCKETS as f32;
    }

    // Remaining dims (13-22) still reserved for future use

    features
}

fn regret_matching(advantages: &[f32]) -> Vec<f32> {
    let positive: Vec<f32> = advantages.iter().map(|&a| a.max(0.0)).collect();
    let sum: f32 = positive.iter().sum();
    if sum > 0.0 {
        positive.iter().map(|&r| r / sum).collect()
    } else {
        // CRITICAL FIX: When all advantages are negative, use argmax to pick
        // the "least bad" action instead of uniform random (which causes
        // suicidal bluffs like All-In with trash hands)
        let mut best_idx = 0;
        let mut best_val = advantages[0];
        for (i, &val) in advantages.iter().enumerate() {
            if val > best_val {
                best_val = val;
                best_idx = i;
            }
        }
        let mut result = vec![0.0; TARGET_DIM];
        if best_idx < TARGET_DIM {
            result[best_idx] = 1.0;
        }
        result
    }
}

// ============================================================================
// Python Interface
// ============================================================================

#[pyclass]
pub struct StepResultFull {
    request_inference: bool,
    finished: bool,
    count: usize,
    samples: usize,
}

#[pymethods]
impl StepResultFull {
    pub fn is_request_inference(&self) -> bool {
        self.request_inference
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn samples(&self) -> usize {
        self.samples
    }
}

#[pyclass]
pub struct ParallelTrainerFull {
    workers: Vec<Worker>,
    query_buffer: Arc<SharedQueryBuffer>,
    prediction_cache: Arc<SharedPredictionCache>,
    writer: Arc<SharedWriter>,
    data_dir: PathBuf,
    current_epoch: usize,
}

#[pymethods]
impl ParallelTrainerFull {
    #[new]
    #[pyo3(signature = (
        data_dir,
        query_buffer_size = 4096,
        num_workers = 8,
        small_blind = 0.5,
        big_blind = 1.0,
        starting_stack = 100.0,
        fixed_flop = None
    ))]
    pub fn new(
        data_dir: String,
        query_buffer_size: usize,
        num_workers: usize,
        small_blind: f64,
        big_blind: f64,
        starting_stack: f64,
        fixed_flop: Option<Vec<u8>>,
    ) -> PyResult<Self> {
        let path = PathBuf::from(&data_dir);
        std::fs::create_dir_all(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create dir: {}", e))
        })?;

        // If fixed_flop provided, all training will use this flop (3 cards)
        // Turn and river will still be random
        let workers: Vec<Worker> = (0..num_workers)
            .map(|_id| Worker::new(small_blind, big_blind, starting_stack, fixed_flop.clone()))
            .collect();

        Ok(Self {
            workers,
            query_buffer: Arc::new(SharedQueryBuffer::new(query_buffer_size)),
            prediction_cache: Arc::new(SharedPredictionCache::new()),
            writer: Arc::new(SharedWriter::new()),
            data_dir: path,
            current_epoch: 0,
        })
    }

    pub fn start_epoch(&mut self, epoch: usize) -> PyResult<()> {
        self.writer.close().ok();
        self.current_epoch = epoch;

        let epoch_path = self.data_dir.join(format!("epoch_{}.bin", epoch));
        let w = TrajectoryWriterFull::new(&epoch_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create file: {}", e))
        })?;
        self.writer.set_writer(w);

        Ok(())
    }

    pub fn end_epoch(&mut self) -> PyResult<usize> {
        self.writer.flush().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(e)
        })
    }

    #[pyo3(signature = (inference_results = None, num_traversals = None))]
    pub fn step(
        &mut self,
        inference_results: Option<&Bound<'_, PyArray2<f32>>>,
        num_traversals: Option<usize>,
    ) -> PyResult<StepResultFull> {
        // Initialize workers with targets
        if let Some(n) = num_traversals {
            let per_worker = n / self.workers.len();
            let remainder = n % self.workers.len();

            for (i, worker) in self.workers.iter_mut().enumerate() {
                let target = per_worker + if i < remainder { 1 } else { 0 };
                worker.set_target(target);
            }

            self.query_buffer.clear();
            self.prediction_cache.clear();
        }

        // Distribute predictions
        if let Some(preds) = inference_results {
            let pred_array = preds.readonly();
            let pred_slice = pred_array.as_slice()?;
            let query_ids = self.query_buffer.get_query_ids();

            for (i, &query_id) in query_ids.iter().enumerate() {
                let offset = i * TARGET_DIM;
                let advantages = &pred_slice[offset..offset + TARGET_DIM];
                let strategy = regret_matching(advantages);
                self.prediction_cache.insert(query_id, strategy);
            }

            self.query_buffer.clear();
        }

        // Run workers until batch is full or all done
        let qb = &self.query_buffer;
        let pc = &self.prediction_cache;
        let writer = &self.writer;

        loop {
            // Check if all workers are done
            let all_done = self.workers.iter().all(|w| w.is_done());
            if all_done {
                return Ok(StepResultFull {
                    request_inference: false,
                    finished: true,
                    count: 0,
                    samples: writer.samples(),
                });
            }

            // Process workers in parallel
            let _: Vec<bool> = self.workers
                .par_iter_mut()
                .map(|worker| {
                    let mut iterations = 0;
                    while iterations < 5000 {
                        if worker.is_done() {
                            break;
                        }
                        if !worker.step(qb, pc, writer) {
                            break;
                        }
                        iterations += 1;

                        if qb.count() >= MIN_BATCH_SIZE {
                            break;
                        }
                    }
                    true
                })
                .collect();

            // Check if batch is ready
            let count = qb.count();
            if count >= MIN_BATCH_SIZE {
                return Ok(StepResultFull {
                    request_inference: true,
                    finished: false,
                    count,
                    samples: writer.samples(),
                });
            }

            // If any queries pending, return for inference
            if count > 0 {
                return Ok(StepResultFull {
                    request_inference: true,
                    finished: false,
                    count,
                    samples: writer.samples(),
                });
            }

            // Check again if all done
            let all_done = self.workers.iter().all(|w| w.is_done());
            if all_done {
                return Ok(StepResultFull {
                    request_inference: false,
                    finished: true,
                    count: 0,
                    samples: writer.samples(),
                });
            }
        }
    }

    pub fn get_query_buffer<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let count = self.query_buffer.count();
        if count == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Empty buffer"));
        }

        let states = self.query_buffer.get_slice();
        let array = ndarray::Array2::from_shape_vec((count, STATE_DIM), states)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        Ok(PyArray2::from_array_bound(py, &array))
    }

    pub fn samples(&self) -> usize {
        self.writer.samples()
    }

    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Returns debug info
    pub fn debug_workers(&self) -> Vec<(usize, usize, usize)> {
        self.workers.iter().map(|w| (w.started, w.completed, w.stack.len())).collect()
    }

    /// Get state and target dimensions
    pub fn get_dimensions(&self) -> (usize, usize) {
        (STATE_DIM, TARGET_DIM)
    }
}
