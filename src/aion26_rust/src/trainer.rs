/// Deep CFR Trainer - Cooperative Inference + Disk-Native Storage
///
/// This module implements the full "Rust Driver" pattern for Deep CFR:
/// - Cooperative step() protocol for batched neural network inference
/// - Disk-native storage via TrajectoryWriter (no forgetting)
/// - Stack machine for pause/resume traversal semantics
///
/// Architecture:
/// 1. Python calls trainer.step(predictions) in a loop
/// 2. Rust traverses game tree, requesting inference when needed
/// 3. Samples written directly to disk (epoch_N.bin files)
/// 4. Python trains from all historical data via memory-mapped access

use pyo3::prelude::*;
use numpy::{PyArray2, PyArrayMethods};
use rand::Rng;
use std::path::PathBuf;
use std::collections::HashMap;
use crate::river::RustRiverHoldem;
use crate::evaluator::{evaluate_7_cards, get_hand_category};
use crate::io::{TrajectoryWriter, STATE_DIM, TARGET_DIM};

// ============================================================================
// State Machine Types
// ============================================================================

/// Traversal stage in the state machine
#[derive(Clone, Debug)]
enum TraversalStage {
    /// Just arrived at this node, need network inference for strategy
    PreInference,

    /// Got strategy from network, exploring actions one by one
    PostInference {
        strategy: Vec<f32>,
        action_values: Vec<f32>,
        action_idx: usize,
    },

    /// All actions explored, ready to compute regrets and write to disk
    Finalizing {
        strategy: Vec<f32>,
        action_values: Vec<f32>,
    },

    /// Chance node waiting for child to return value (then propagates up)
    WaitingForChild,
}

/// Context for a single traversal node (stack frame)
#[derive(Clone)]
struct TraversalContext {
    game_state: RustRiverHoldem,
    update_player: u8,
    reach_prob_0: f32,
    reach_prob_1: f32,
    current_player: i8,
    legal_actions: Vec<u8>,
    stage: TraversalStage,
    query_id: Option<usize>,
    /// For chance/opponent nodes: stores returned value from child
    child_value: Option<f32>,
}

// ============================================================================
// Query Buffer
// ============================================================================

/// Minimum batch size before returning to Python for inference
/// This prevents ping-ponging single queries to the GPU
/// Set to 2048 to maximize GPU utilization (keep GPU residency >80%)
const MIN_BATCH_SIZE: usize = 2048;

/// Maximum number of in-flight traversals to prevent memory exhaustion
/// Each traversal needs multiple contexts for the game tree
/// Set high enough that we can fill the batch (each traversal adds ~1 query initially)
const MAX_IN_FLIGHT_TRAVERSALS: usize = 32768;

/// Query buffer for batched network inference
struct QueryBuffer {
    states: Vec<f32>,         // Flattened: max_queries × 136 floats
    query_ids: Vec<usize>,    // Query IDs for matching predictions to contexts
    max_queries: usize,
    count: usize,
    next_query_id: usize,
}

impl QueryBuffer {
    fn new(max_queries: usize) -> Self {
        Self {
            states: vec![0.0; max_queries * STATE_DIM],
            query_ids: Vec::with_capacity(max_queries),
            max_queries,
            count: 0,
            next_query_id: 0,
        }
    }

    fn add_query(&mut self, state_encoding: &[f32]) -> Option<usize> {
        if self.count >= self.max_queries {
            return None;
        }

        debug_assert_eq!(state_encoding.len(), STATE_DIM);

        let offset = self.count * STATE_DIM;
        self.states[offset..offset + STATE_DIM].copy_from_slice(state_encoding);

        let query_id = self.next_query_id;
        self.query_ids.push(query_id);
        self.next_query_id += 1;
        self.count += 1;

        Some(query_id)
    }

    fn clear(&mut self) {
        self.count = 0;
        self.query_ids.clear();
    }

    fn as_slice(&self) -> &[f32] {
        &self.states[0..self.count * STATE_DIM]
    }
}

// ============================================================================
// State Encoder
// ============================================================================

struct StateEncoder;

impl StateEncoder {
    fn encode(state: &RustRiverHoldem, player: u8) -> Vec<f32> {
        let mut features = vec![0.0; STATE_DIM];

        if !state.is_dealt {
            return features;
        }

        let hand = &state.hands[player as usize];
        let board = &state.board;

        // 1. Hand rank category (10 dims)
        let seven_cards: Vec<u8> = hand.iter().chain(board.iter()).cloned().collect();
        if seven_cards.len() == 7 {
            let rank = evaluate_7_cards(&seven_cards.try_into().unwrap());
            let category = get_hand_category(rank) as usize;
            features[category] = 1.0;
        }

        // 2. Hole cards (34 dims: 2 cards × 17 bits)
        for (i, &card) in hand.iter().enumerate() {
            let offset = 10 + i * 17;
            Self::encode_card_onehot(card, &mut features[offset..offset + 17]);
        }

        // 3. Board cards (85 dims: 5 cards × 17 bits)
        for (i, &card) in board.iter().enumerate() {
            let offset = 10 + 34 + i * 17;
            Self::encode_card_onehot(card, &mut features[offset..offset + 17]);
        }

        // 4. Betting context (7 dims)
        let context_offset = 10 + 34 + 85;
        let max_pot = 500.0;
        let max_stack = 200.0;

        let current_invested = if player == 0 {
            state.player_0_invested
        } else {
            state.player_1_invested
        };

        let call_amount = (state.current_bet - current_invested).max(0.0);
        let pot_after_call = state.pot + call_amount;
        let pot_odds = if pot_after_call > 0.0 {
            call_amount / pot_after_call
        } else {
            0.0
        };

        features[context_offset] = (state.pot / max_pot) as f32;
        features[context_offset + 1] = (state.stacks[0] / max_stack) as f32;
        features[context_offset + 2] = (state.stacks[1] / max_stack) as f32;
        features[context_offset + 3] = (state.current_bet / max_stack) as f32;
        features[context_offset + 4] = (state.player_0_invested / max_stack) as f32;
        features[context_offset + 5] = (state.player_1_invested / max_stack) as f32;
        features[context_offset + 6] = pot_odds as f32;

        features
    }

    fn encode_card_onehot(card: u8, output: &mut [f32]) {
        let rank = card % 13;
        let suit = card / 13;
        output[rank as usize] = 1.0;
        output[13 + suit as usize] = 1.0;
    }
}

// ============================================================================
// Step Result (Python Interface)
// ============================================================================

/// Result of calling step() - tells Python what to do next
#[pyclass]
#[derive(Clone, Debug)]
pub enum StepResult {
    /// Need inference: Rust has `count` queries waiting in buffer
    RequestInference { count: usize },

    /// Batch finished: `samples_added` samples written to disk
    Finished { samples_added: usize },
}

#[pymethods]
impl StepResult {
    fn is_request_inference(&self) -> bool {
        matches!(self, StepResult::RequestInference { .. })
    }

    fn is_finished(&self) -> bool {
        matches!(self, StepResult::Finished { .. })
    }

    fn count(&self) -> usize {
        match self {
            StepResult::RequestInference { count } => *count,
            StepResult::Finished { samples_added } => *samples_added,
        }
    }

    fn __repr__(&self) -> String {
        match self {
            StepResult::RequestInference { count } => format!("RequestInference(count={})", count),
            StepResult::Finished { samples_added } => format!("Finished(samples_added={})", samples_added),
        }
    }
}

// ============================================================================
// RustTrainer - The Main PyClass
// ============================================================================

/// Rust-driven Deep CFR Trainer with Cooperative Inference + Disk Storage
///
/// Usage (Python):
/// ```python
/// trainer = RustTrainer("data/trajectories")
/// for epoch in range(1000):
///     trainer.start_epoch(epoch)
///     predictions = None
///     while True:
///         result = trainer.step(predictions, num_traversals=100)
///         if result.is_finished():
///             break
///         # GPU inference
///         queries = trainer.get_query_buffer()
///         predictions = network(queries).numpy()
///     trainer.end_epoch()
/// ```
#[pyclass]
pub struct RustTrainer {
    // Game configuration
    initial_game: RustRiverHoldem,

    // Disk storage
    data_dir: PathBuf,
    current_epoch: usize,
    writer: Option<TrajectoryWriter>,
    epoch_samples: usize,
    total_samples: usize,

    // Query buffer for batched inference
    query_buffer: QueryBuffer,

    // Stack machine for traversal
    context_stack: Vec<TraversalContext>,

    // Traversal tracking
    completed_traversals: usize,
    started_traversals: usize,  // Track how many traversals we've started
    target_traversals: usize,
    iteration: usize,

    // Prediction cache (query_id -> strategy)
    prediction_cache: HashMap<usize, Vec<f32>>,

    // RNG
    rng: rand::rngs::StdRng,

    // DEBUG counters
    debug_add_query_calls: usize,
    debug_waiting_calls: usize,
}

#[pymethods]
impl RustTrainer {
    #[new]
    #[pyo3(signature = (data_dir, query_buffer_size = 4096, fixed_board = None))]
    pub fn new(
        data_dir: String,
        query_buffer_size: usize,
        fixed_board: Option<Vec<u8>>,
    ) -> PyResult<Self> {
        let path = PathBuf::from(&data_dir);

        // Create data directory if it doesn't exist
        std::fs::create_dir_all(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to create data directory: {}", e)
            )
        })?;

        Ok(Self {
            initial_game: RustRiverHoldem::new(
                vec![100.0, 100.0],
                2.0,
                0.0,
                1.0,
                1.0,
                fixed_board,
            ),
            data_dir: path,
            current_epoch: 0,
            writer: None,
            epoch_samples: 0,
            total_samples: 0,
            query_buffer: QueryBuffer::new(query_buffer_size),
            context_stack: Vec::new(),
            completed_traversals: 0,
            started_traversals: 0,
            target_traversals: 0,
            iteration: 0,
            prediction_cache: HashMap::new(),
            rng: rand::SeedableRng::from_entropy(),
            debug_add_query_calls: 0,
            debug_waiting_calls: 0,
        })
    }

    // ========================================================================
    // Epoch Management
    // ========================================================================

    /// Start a new epoch for data collection
    pub fn start_epoch(&mut self, epoch: usize) -> PyResult<()> {
        // Close previous writer if any
        if let Some(mut writer) = self.writer.take() {
            writer.flush().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(
                    format!("Failed to flush previous epoch: {}", e)
                )
            })?;
        }

        self.current_epoch = epoch;
        self.epoch_samples = 0;
        let epoch_path = self.data_dir.join(format!("epoch_{}.bin", epoch));

        self.writer = Some(TrajectoryWriter::new(&epoch_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to create epoch file: {}", e)
            )
        })?);

        Ok(())
    }

    /// End the current epoch and flush to disk
    pub fn end_epoch(&mut self) -> PyResult<usize> {
        if let Some(mut writer) = self.writer.take() {
            let samples = writer.len();
            writer.flush().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(
                    format!("Failed to flush epoch: {}", e)
                )
            })?;
            Ok(samples)
        } else {
            Ok(0)
        }
    }

    // ========================================================================
    // Cooperative Step (The Heart of the System)
    // ========================================================================

    /// Execute one step of cooperative traversal
    ///
    /// Args:
    ///     inference_results: Predictions from previous RequestInference (shape: [N, 4])
    ///     num_traversals: Number of traversals to complete (required on first call)
    ///
    /// Returns:
    ///     StepResult::RequestInference if Rust needs network predictions
    ///     StepResult::Finished when all traversals complete
    #[pyo3(signature = (inference_results = None, num_traversals = None))]
    pub fn step(
        &mut self,
        inference_results: Option<&Bound<'_, PyArray2<f32>>>,
        num_traversals: Option<usize>,
    ) -> PyResult<StepResult> {
        // Ensure we have an active writer
        if self.writer.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "No active epoch. Call start_epoch() first."
            ));
        }

        // Initialize batch if this is the first step
        if let Some(n) = num_traversals {
            self.target_traversals = n;
            self.completed_traversals = 0;
            self.started_traversals = 0;
            self.context_stack.clear();
            self.prediction_cache.clear();
        }

        // Distribute inference results to waiting contexts
        if let Some(predictions) = inference_results {
            self.distribute_predictions(predictions)?;
        }

        // Process contexts until we need inference or finish
        // "Fill the Boat" logic: accumulate queries before returning to Python
        loop {
            // Calculate in-flight traversals (started but not completed)
            let in_flight = self.started_traversals - self.completed_traversals;

            // Try to resume pending contexts
            if !self.context_stack.is_empty() {
                match self.resume_top_context()? {
                    ResumeResult::NeedInference => {
                        // Check if batch is full - if so, return for inference
                        if self.query_buffer.count >= MIN_BATCH_SIZE {
                            return Ok(StepResult::RequestInference {
                                count: self.query_buffer.count,
                            });
                        }

                        // Try to start a new traversal to fill the batch
                        // But limit in-flight to prevent memory exhaustion
                        if self.started_traversals < self.target_traversals
                            && in_flight < MAX_IN_FLIGHT_TRAVERSALS
                        {
                            self.start_new_traversal()?;
                            continue;
                        }

                        // Can't start more traversals and batch not full yet.
                        // Try to find another context that can make progress.
                        // Search backwards through the stack for a context with
                        // a cached prediction that we can advance.
                        let mut found_ready = false;
                        for idx in (0..self.context_stack.len() - 1).rev() {
                            if let Some(query_id) = self.context_stack[idx].query_id {
                                if self.prediction_cache.contains_key(&query_id) {
                                    // This context has a prediction - swap it to top
                                    let last = self.context_stack.len() - 1;
                                    self.context_stack.swap(idx, last);
                                    found_ready = true;
                                    break;
                                }
                            }
                        }

                        if found_ready {
                            continue;  // Try resuming the newly swapped context
                        }

                        // No ready contexts found - return with what we have
                        if self.query_buffer.count > 0 {
                            return Ok(StepResult::RequestInference {
                                count: self.query_buffer.count,
                            });
                        }

                        // Edge case: context needs inference but buffer empty
                        // This shouldn't happen, but avoid infinite loop
                        break;
                    }
                    ResumeResult::MadeProgress => continue,
                    ResumeResult::ContextPopped => continue,
                }
            }

            // No pending contexts, start new traversal if needed
            if self.started_traversals < self.target_traversals {
                self.start_new_traversal()?;
                continue;
            }

            // All traversals complete
            break;
        }

        Ok(StepResult::Finished {
            samples_added: self.epoch_samples,
        })
    }

    /// Get query buffer for network inference (zero-copy view)
    pub fn get_query_buffer<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let count = self.query_buffer.count;
        if count == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Query buffer is empty"
            ));
        }

        let slice = self.query_buffer.as_slice();

        // Create a view into Rust's memory
        let array = ndarray::ArrayView2::from_shape((count, STATE_DIM), slice)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to create array view: {}", e)
            ))?;

        // Convert to PyArray (this copies, but it's safe)
        Ok(PyArray2::from_array_bound(py, &array))
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    pub fn epoch_samples(&self) -> usize {
        self.epoch_samples
    }

    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    pub fn iteration(&self) -> usize {
        self.iteration
    }

    pub fn data_dir(&self) -> String {
        self.data_dir.to_string_lossy().to_string()
    }

    pub fn query_buffer_count(&self) -> usize {
        self.query_buffer.count
    }

    pub fn pending_contexts(&self) -> usize {
        self.context_stack.len()
    }

    pub fn completed_traversals(&self) -> usize {
        self.completed_traversals
    }

    pub fn started_traversals(&self) -> usize {
        self.started_traversals
    }

    pub fn target_traversals(&self) -> usize {
        self.target_traversals
    }

    pub fn debug_add_query_calls(&self) -> usize {
        self.debug_add_query_calls
    }

    pub fn debug_waiting_calls(&self) -> usize {
        self.debug_waiting_calls
    }
}

// ============================================================================
// Private Implementation
// ============================================================================

enum ResumeResult {
    NeedInference,
    MadeProgress,
    ContextPopped,
}

impl RustTrainer {
    /// Distribute predictions to waiting contexts
    fn distribute_predictions(&mut self, predictions: &Bound<'_, PyArray2<f32>>) -> PyResult<()> {
        let pred_array = predictions.readonly();
        let pred_slice = pred_array.as_slice()?;

        // DON'T clear the entire cache - other contexts may still need their strategies
        // Just insert/overwrite the new predictions
        for (i, &query_id) in self.query_buffer.query_ids.iter().enumerate() {
            let offset = i * TARGET_DIM;
            let advantages = &pred_slice[offset..offset + TARGET_DIM];

            // Apply regret matching to convert advantages to strategy
            let strategy = self.regret_matching(advantages);
            self.prediction_cache.insert(query_id, strategy);
        }

        self.query_buffer.clear();
        Ok(())
    }

    /// Regret matching: convert advantages to probabilities
    fn regret_matching(&self, advantages: &[f32]) -> Vec<f32> {
        let positive_regrets: Vec<f32> = advantages.iter().map(|&a| a.max(0.0)).collect();
        let sum: f32 = positive_regrets.iter().sum();

        if sum > 0.0 {
            positive_regrets.iter().map(|&r| r / sum).collect()
        } else {
            // CRITICAL FIX: When all advantages are negative, use argmax to pick
            // the "least bad" action instead of uniform random
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

    /// Start a new traversal
    fn start_new_traversal(&mut self) -> PyResult<()> {
        self.iteration += 1;
        self.started_traversals += 1;

        // Alternate between players
        let update_player = (self.started_traversals % 2) as u8;

        let ctx = self.create_initial_context(
            self.initial_game.clone(),
            update_player,
            1.0,
            1.0,
        )?;

        self.context_stack.push(ctx);
        Ok(())
    }

    /// Create initial context for a state
    fn create_initial_context(
        &self,
        game_state: RustRiverHoldem,
        update_player: u8,
        reach_prob_0: f32,
        reach_prob_1: f32,
    ) -> PyResult<TraversalContext> {
        let current_player = game_state.current_player();
        let legal_actions = game_state.legal_actions();

        Ok(TraversalContext {
            game_state,
            update_player,
            reach_prob_0,
            reach_prob_1,
            current_player,
            legal_actions,
            stage: TraversalStage::PreInference,
            query_id: None,
            child_value: None,
        })
    }

    /// Resume the top context on the stack
    fn resume_top_context(&mut self) -> PyResult<ResumeResult> {
        let idx = self.context_stack.len() - 1;

        // Handle terminal nodes immediately
        if self.context_stack[idx].game_state.is_terminal() {
            let value = self.handle_terminal_node(idx)?;
            self.propagate_value_and_pop(idx, value)?;
            return Ok(ResumeResult::ContextPopped);
        }

        // Handle state machine stages
        match self.context_stack[idx].stage.clone() {
            TraversalStage::WaitingForChild => {
                // Child has returned, propagate value up
                let value = self.context_stack[idx].child_value.unwrap_or(0.0);
                self.propagate_value_and_pop(idx, value)?;
                Ok(ResumeResult::ContextPopped)
            }
            TraversalStage::PreInference => {
                // Handle chance nodes (deal cards) - only in PreInference
                if self.context_stack[idx].game_state.is_chance_node() {
                    self.handle_chance_node(idx)?;
                    return Ok(ResumeResult::MadeProgress);
                }
                self.handle_pre_inference(idx)
            }
            TraversalStage::PostInference { strategy, action_values, action_idx } => {
                self.handle_post_inference(idx, strategy, action_values, action_idx)
            }
            TraversalStage::Finalizing { strategy, action_values } => {
                self.handle_finalizing(idx, strategy, action_values)
            }
        }
    }

    /// Handle terminal node
    fn handle_terminal_node(&self, idx: usize) -> PyResult<f32> {
        let ctx = &self.context_stack[idx];
        let returns = ctx.game_state.returns();
        Ok(returns[ctx.update_player as usize] as f32)
    }

    /// Handle chance node (deal cards)
    fn handle_chance_node(&mut self, idx: usize) -> PyResult<()> {
        let ctx = &self.context_stack[idx];
        let next_state = ctx.game_state.apply_action(0).expect("Failed to deal cards");

        let child_ctx = self.create_initial_context(
            next_state,
            ctx.update_player,
            ctx.reach_prob_0,
            ctx.reach_prob_1,
        )?;

        // Transition chance node to WaitingForChild stage
        self.context_stack[idx].stage = TraversalStage::WaitingForChild;

        self.context_stack.push(child_ctx);
        Ok(())
    }

    /// Handle PreInference stage: need network prediction
    fn handle_pre_inference(&mut self, idx: usize) -> PyResult<ResumeResult> {
        let ctx = &self.context_stack[idx];
        let is_update_player = ctx.current_player as u8 == ctx.update_player;

        // Check if we already have prediction cached
        if let Some(query_id) = ctx.query_id {
            if let Some(strategy) = self.prediction_cache.get(&query_id).cloned() {
                if is_update_player {
                    // Update player: explore ALL actions (compute counterfactual values)
                    let num_actions = ctx.legal_actions.len();
                    let action_values = vec![0.0; num_actions];

                    self.context_stack[idx].stage = TraversalStage::PostInference {
                        strategy,
                        action_values,
                        action_idx: 0,
                    };
                } else {
                    // Opponent: sample ONE action (External Sampling MCCFR)
                    // Still go through PostInference but with sampled_action_idx set to skip to that action
                    let num_actions = ctx.legal_actions.len();
                    let sampled_idx = self.sample_action(&strategy);

                    // Set action_idx to the sampled action so PostInference explores only that one
                    self.context_stack[idx].stage = TraversalStage::PostInference {
                        strategy,
                        action_values: vec![0.0; num_actions],
                        action_idx: sampled_idx,  // Start at sampled action
                    };
                }

                return Ok(ResumeResult::MadeProgress);
            } else {
                // Already have pending query but no strategy yet.
                self.debug_waiting_calls += 1;
                if self.query_buffer.count > 0 {
                    return Ok(ResumeResult::NeedInference);
                } else {
                    // This shouldn't happen (we have query_id but empty buffer)
                    return Ok(ResumeResult::MadeProgress);
                }
            }
        }

        // Need to request inference - no query_id yet
        let state_encoding = StateEncoder::encode(&ctx.game_state, ctx.update_player);

        if let Some(query_id) = self.query_buffer.add_query(&state_encoding) {
            self.debug_add_query_calls += 1;
            self.context_stack[idx].query_id = Some(query_id);
            // CRITICAL: Return NeedInference immediately after adding query
            // This prevents the loop from adding duplicate queries
            Ok(ResumeResult::NeedInference)
        } else {
            // Buffer full, need inference now
            Ok(ResumeResult::NeedInference)
        }
    }

    /// Handle PostInference stage: explore actions
    fn handle_post_inference(
        &mut self,
        idx: usize,
        strategy: Vec<f32>,
        action_values: Vec<f32>,
        action_idx: usize,
    ) -> PyResult<ResumeResult> {
        let ctx = &self.context_stack[idx];
        let is_update_player = ctx.current_player as u8 == ctx.update_player;

        // For opponent: only explore ONE action (the sampled one)
        // For update player: explore ALL actions
        let should_explore = if is_update_player {
            action_idx < ctx.legal_actions.len()
        } else {
            // Opponent: only explore if this is the first (sampled) action
            action_idx < ctx.legal_actions.len() &&
            action_idx == self.get_sampled_action_idx(&ctx.legal_actions, &strategy, action_idx)
        };

        if should_explore {
            // Explore this action
            let action = ctx.legal_actions[action_idx];
            let next_state = ctx.game_state.apply_action(action).expect("Failed to apply action");

            let new_reach_0 = if ctx.current_player == 0 {
                ctx.reach_prob_0 * strategy.get(action_idx).copied().unwrap_or(0.25)
            } else {
                ctx.reach_prob_0
            };

            let new_reach_1 = if ctx.current_player == 1 {
                ctx.reach_prob_1 * strategy.get(action_idx).copied().unwrap_or(0.25)
            } else {
                ctx.reach_prob_1
            };

            // Create child context
            let child_ctx = self.create_initial_context(
                next_state,
                ctx.update_player,
                new_reach_0,
                new_reach_1,
            )?;

            // For opponent: jump straight to Finalizing after exploring one action
            // For update player: continue to next action
            if is_update_player {
                self.context_stack[idx].stage = TraversalStage::PostInference {
                    strategy,
                    action_values,
                    action_idx: action_idx + 1,
                };
            } else {
                // Opponent: after this child completes, go to Finalizing
                self.context_stack[idx].stage = TraversalStage::PostInference {
                    strategy,
                    action_values,
                    action_idx: ctx.legal_actions.len(),  // Force exit on next iteration
                };
            }

            self.context_stack.push(child_ctx);
            Ok(ResumeResult::MadeProgress)
        } else {
            // All actions explored (or opponent done), transition to Finalizing
            self.context_stack[idx].stage = TraversalStage::Finalizing {
                strategy,
                action_values,
            };
            Ok(ResumeResult::MadeProgress)
        }
    }

    /// Helper to check if this is the sampled action for opponent nodes
    fn get_sampled_action_idx(&self, _legal_actions: &[u8], _strategy: &[f32], current_idx: usize) -> usize {
        // For opponent nodes, action_idx was set to the sampled index in PreInference
        // So just check if we're at that index
        current_idx
    }

    /// Handle Finalizing stage: compute regrets and write to disk
    ///
    /// For update player: compute regrets and write sample
    /// For opponent: just propagate the sampled value (no sample written)
    fn handle_finalizing(
        &mut self,
        idx: usize,
        strategy: Vec<f32>,
        action_values: Vec<f32>,
    ) -> PyResult<ResumeResult> {
        let ctx = &self.context_stack[idx];
        let is_update_player = ctx.current_player as u8 == ctx.update_player;

        if is_update_player {
            // Update player: compute regrets and write sample

            // Compute expected value
            let expected_value: f32 = action_values.iter()
                .zip(strategy.iter())
                .map(|(v, s)| v * s)
                .sum();

            // Compute instant regrets
            let mut regrets = [0.0f32; TARGET_DIM];
            for i in 0..action_values.len().min(TARGET_DIM) {
                regrets[i] = action_values[i] - expected_value;
            }

            // Encode state
            let state_encoding = StateEncoder::encode(&ctx.game_state, ctx.update_player);
            let state_arr: [f32; STATE_DIM] = state_encoding.try_into()
                .expect("State encoding has wrong dimension");

            // Write directly to disk
            if let Some(ref mut writer) = self.writer {
                writer.append(&state_arr, &regrets).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(
                        format!("Failed to write sample: {}", e)
                    )
                })?;
                self.epoch_samples += 1;
                self.total_samples += 1;
            }

            // Propagate expected value up
            self.propagate_value_and_pop(idx, expected_value)?;
        } else {
            // Opponent: just propagate the sampled action's value
            // action_values should have one value from the sampled child
            let value = action_values.iter().find(|&&v| v != 0.0).copied().unwrap_or(0.0);
            self.propagate_value_and_pop(idx, value)?;
        }

        Ok(ResumeResult::ContextPopped)
    }

    /// Propagate value to parent and pop context
    fn propagate_value_and_pop(&mut self, idx: usize, value: f32) -> PyResult<()> {
        if idx == 0 {
            // Root context, traversal complete
            self.completed_traversals += 1;
            self.context_stack.pop();
            return Ok(());
        }

        let parent_idx = idx - 1;

        // Update parent based on its stage
        match &mut self.context_stack[parent_idx].stage {
            TraversalStage::PostInference { ref mut action_values, action_idx, .. } => {
                if *action_idx > 0 && *action_idx - 1 < action_values.len() {
                    action_values[*action_idx - 1] = value;
                }
            }
            TraversalStage::WaitingForChild => {
                // Store child value for chance/opponent nodes
                self.context_stack[parent_idx].child_value = Some(value);
            }
            _ => {
                // Other stages don't expect child values
            }
        }

        self.context_stack.pop();
        Ok(())
    }

    /// Sample an action according to strategy
    fn sample_action(&mut self, strategy: &[f32]) -> usize {
        let r: f32 = self.rng.gen();
        let mut cumulative = 0.0;
        for (i, &prob) in strategy.iter().enumerate() {
            cumulative += prob;
            if r < cumulative {
                return i;
            }
        }
        strategy.len().saturating_sub(1)
    }
}
