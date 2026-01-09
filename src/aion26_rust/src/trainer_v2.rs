/// Deep CFR Trainer V2 - Stack Machine Architecture with Cooperative Inference
///
/// This version implements a state machine to support pause/resume semantics
/// required for the batch inference protocol.

use pyo3::prelude::*;
use numpy::{PyArray2, PyArrayMethods};
use rand::Rng;
use crate::river::RustRiverHoldem;
use crate::evaluator::{evaluate_7_cards, get_hand_category};

/// Traversal stage in the state machine
#[derive(Clone, Debug)]
enum TraversalStage {
    /// Just arrived at this node, need network inference
    PreInference,

    /// Got strategy from network, exploring actions
    PostInference {
        strategy: Vec<f32>,
        action_values: Vec<f32>,
        action_idx: usize,  // Which action we're currently exploring
    },

    /// All actions explored, computing final value
    Finalizing {
        strategy: Vec<f32>,
        action_values: Vec<f32>,
    },
}

/// Context for a single traversal node (stack frame)
#[derive(Clone, Debug)]
struct TraversalContext {
    /// Game state at this node
    game_state: RustRiverHoldem,

    /// Player being updated (0 or 1)
    update_player: u8,

    /// Reach probability for player 0
    reach_prob_0: f32,

    /// Reach probability for player 1
    reach_prob_1: f32,

    /// Current player at this node (-1 for terminal/chance)
    current_player: i8,

    /// Legal actions at this node
    legal_actions: Vec<u8>,

    /// Current stage in traversal
    stage: TraversalStage,

    /// Query ID if waiting for inference (for matching predictions)
    query_id: Option<usize>,

    /// Parent context index (for stack management)
    parent_idx: Option<usize>,
}

/// Result of attempting to continue a traversal
#[derive(Debug)]
enum ContinueResult {
    /// Traversal completed, returned a value
    Completed(f32),

    /// Need inference, paused execution
    NeedInference,

    /// Need to recurse into child node
    Recurse {
        child_state: RustRiverHoldem,
        child_update_player: u8,
        child_reach_0: f32,
        child_reach_1: f32,
    },
}

/// Query buffer for batched network inference
struct QueryBuffer {
    /// Flattened state encodings: max_queries × 136 floats
    states: Vec<f32>,

    /// Query IDs for matching predictions to contexts
    query_ids: Vec<usize>,

    /// Maximum number of queries
    max_queries: usize,

    /// Current number of queries
    count: usize,

    /// Next query ID to assign
    next_query_id: usize,
}

impl QueryBuffer {
    fn new(max_queries: usize) -> Self {
        Self {
            states: vec![0.0; max_queries * 136],
            query_ids: Vec::with_capacity(max_queries),
            max_queries,
            count: 0,
            next_query_id: 0,
        }
    }

    /// Add a query to the buffer
    ///
    /// Returns: Some(query_id) if added, None if buffer full
    fn add_query(&mut self, state_encoding: &[f32]) -> Option<usize> {
        if self.count >= self.max_queries {
            return None;  // Buffer full
        }

        debug_assert_eq!(state_encoding.len(), 136);

        let offset = self.count * 136;
        self.states[offset..offset + 136].copy_from_slice(state_encoding);

        let query_id = self.next_query_id;
        self.query_ids.push(query_id);
        self.next_query_id += 1;
        self.count += 1;

        Some(query_id)
    }

    /// Clear the buffer (after predictions received)
    fn clear(&mut self) {
        self.count = 0;
        self.query_ids.clear();
    }

    /// Get the current state buffer as a slice
    fn as_slice(&self) -> &[f32] {
        &self.states[0..self.count * 136]
    }
}

/// Reservoir buffer (from original implementation)
struct ReservoirBuffer {
    states: Vec<f32>,
    targets: Vec<f32>,
    capacity: usize,
    state_dim: usize,
    target_dim: usize,
    size: usize,
    total_seen: usize,
    rng: rand::rngs::StdRng,
}

impl ReservoirBuffer {
    fn new(capacity: usize, state_dim: usize, target_dim: usize) -> Self {
        Self {
            states: vec![0.0; capacity * state_dim],
            targets: vec![0.0; capacity * target_dim],
            capacity,
            state_dim,
            target_dim,
            size: 0,
            total_seen: 0,
            rng: rand::SeedableRng::from_entropy(),
        }
    }

    fn add(&mut self, state: &[f32], target: &[f32]) {
        debug_assert_eq!(state.len(), self.state_dim);
        debug_assert_eq!(target.len(), self.target_dim);

        self.total_seen += 1;

        if self.size < self.capacity {
            let idx = self.size;
            let state_offset = idx * self.state_dim;
            let target_offset = idx * self.target_dim;

            self.states[state_offset..state_offset + self.state_dim].copy_from_slice(state);
            self.targets[target_offset..target_offset + self.target_dim].copy_from_slice(target);

            self.size += 1;
        } else {
            let j = self.rng.gen_range(0..self.total_seen);
            if j < self.capacity {
                let state_offset = j * self.state_dim;
                let target_offset = j * self.target_dim;

                self.states[state_offset..state_offset + self.state_dim].copy_from_slice(state);
                self.targets[target_offset..target_offset + self.target_dim].copy_from_slice(target);
            }
        }
    }

    fn sample_indices(&mut self, batch_size: usize) -> Vec<usize> {
        (0..batch_size)
            .map(|_| self.rng.gen_range(0..self.size))
            .collect()
    }

    fn len(&self) -> usize {
        self.size
    }

    fn fill_percentage(&self) -> f32 {
        (self.size as f32 / self.capacity as f32) * 100.0
    }
}

/// State encoder (from original implementation)
struct StateEncoder;

impl StateEncoder {
    fn encode(state: &RustRiverHoldem, player: u8) -> Vec<f32> {
        let mut features = vec![0.0; 136];

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

        // 2. Hole cards (34 dims)
        for (i, &card) in hand.iter().enumerate() {
            let offset = 10 + i * 17;
            Self::encode_card_onehot(card, &mut features[offset..offset + 17]);
        }

        // 3. Board cards (85 dims)
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

/// Step result for cooperative protocol
#[pyclass]
#[derive(Clone, Debug)]
pub enum StepResult {
    /// Need inference: Rust has `count` queries waiting
    RequestInference { count: usize },

    /// Batch finished: `samples_added` samples added to replay buffer
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
}

/// Rust-driven Deep CFR Trainer V2 with Stack Machine
#[pyclass]
pub struct RustTrainerV2 {
    /// Initial game configuration
    initial_game: RustRiverHoldem,

    /// Replay buffer
    buffer: ReservoirBuffer,

    /// Query buffer for batched inference
    query_buffer: QueryBuffer,

    /// Stack of pending contexts
    context_stack: Vec<TraversalContext>,

    /// Current number of completed traversals
    completed_traversals: usize,

    /// Target number of traversals for this batch
    target_traversals: usize,

    /// RNG for sampling
    rng: rand::rngs::StdRng,

    /// Iteration counter
    iteration: usize,

    /// Predictions from last inference (query_id -> strategy)
    prediction_cache: std::collections::HashMap<usize, Vec<f32>>,
}

#[pymethods]
impl RustTrainerV2 {
    #[new]
    #[pyo3(signature = (buffer_capacity = 2_000_000, query_buffer_size = 4096, fixed_board = None))]
    pub fn new(
        buffer_capacity: usize,
        query_buffer_size: usize,
        fixed_board: Option<Vec<u8>>,
    ) -> Self {
        Self {
            initial_game: RustRiverHoldem::new(
                vec![100.0, 100.0],
                2.0,
                0.0,
                1.0,
                1.0,
                fixed_board,
            ),
            buffer: ReservoirBuffer::new(buffer_capacity, 136, 4),
            query_buffer: QueryBuffer::new(query_buffer_size),
            context_stack: Vec::new(),
            completed_traversals: 0,
            target_traversals: 0,
            rng: rand::SeedableRng::from_entropy(),
            iteration: 0,
            prediction_cache: std::collections::HashMap::new(),
        }
    }

    /// Cooperative traversal step
    ///
    /// Args:
    ///     inference_results: Predictions from previous RequestInference (shape: [N, 4])
    ///     num_traversals: Number of traversals to complete (only on first call)
    ///
    /// Returns:
    ///     StepResult enum
    pub fn step(
        &mut self,
        inference_results: Option<&Bound<'_, PyArray2<f32>>>,
        num_traversals: Option<usize>,
    ) -> PyResult<StepResult> {
        // Initialize batch if this is the first step
        if let Some(n) = num_traversals {
            self.target_traversals = n;
            self.completed_traversals = 0;
            self.context_stack.clear();
        }

        // Distribute inference results
        if let Some(predictions) = inference_results {
            self.distribute_predictions(predictions)?;
        }

        // Process contexts until we hit a pause point or finish
        loop {
            // Try to resume pending contexts
            if !self.context_stack.is_empty() {
                if let Some(needs_inference) = self.resume_top_context()? {
                    if needs_inference {
                        return Ok(StepResult::RequestInference {
                            count: self.query_buffer.count,
                        });
                    }
                }
                continue;  // Context modified, loop again
            }

            // No pending contexts, start new traversal if needed
            if self.completed_traversals < self.target_traversals {
                self.start_new_traversal()?;
                continue;
            }

            // All traversals complete
            break;
        }

        let samples_added = self.completed_traversals * 2;  // 2 players
        Ok(StepResult::Finished { samples_added })
    }

    /// Get query buffer for network inference (zero-copy view)
    pub fn get_query_buffer<'py>(&self, py: Python<'py>) -> &Bound<'py, PyArray2<f32>> {
        let count = self.query_buffer.count;
        let slice = self.query_buffer.as_slice();

        // SAFETY: slice is valid for entire GIL scope
        unsafe {
            PyArray2::borrow_from_array_bound(
                &ndarray::ArrayView2::from_shape((count, 136), slice).unwrap(),
                py,
            )
        }
    }

    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    pub fn buffer_fill_percentage(&self) -> f32 {
        self.buffer.fill_percentage()
    }

    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Fill batch arrays (for training)
    pub fn fill_batch(
        &mut self,
        states: &Bound<'_, PyArray2<f32>>,
        targets: &Bound<'_, PyArray2<f32>>,
        batch_size: usize,
    ) -> PyResult<()> {
        let indices = self.buffer.sample_indices(batch_size);

        let mut states_array = unsafe { states.as_array_mut() };
        let mut targets_array = unsafe { targets.as_array_mut() };

        for (batch_idx, &buffer_idx) in indices.iter().enumerate() {
            let state_offset = buffer_idx * 136;
            let target_offset = buffer_idx * 4;

            for j in 0..136 {
                states_array[[batch_idx, j]] = self.buffer.states[state_offset + j];
            }

            for j in 0..4 {
                targets_array[[batch_idx, j]] = self.buffer.targets[target_offset + j];
            }
        }

        Ok(())
    }
}

/// Private implementation
impl RustTrainerV2 {
    /// Distribute predictions to waiting contexts
    fn distribute_predictions(&mut self, predictions: &Bound<'_, PyArray2<f32>>) -> PyResult<()> {
        let pred_array = predictions.readonly();
        let pred_slice = pred_array.as_slice()?;

        // Map query_id -> prediction
        self.prediction_cache.clear();
        for (i, &query_id) in self.query_buffer.query_ids.iter().enumerate() {
            let offset = i * 4;
            let strategy = pred_slice[offset..offset + 4].to_vec();

            // Apply regret matching to convert advantages to strategy
            let strategy = self.regret_matching(&strategy);

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
            vec![0.25; 4]  // Uniform fallback
        }
    }

    /// Start a new traversal
    fn start_new_traversal(&mut self) -> PyResult<()> {
        self.iteration += 1;

        let update_player = (self.completed_traversals % 2) as u8;

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
            parent_idx: None,
        })
    }

    /// Resume the top context on the stack
    ///
    /// Returns: Some(true) if needs inference, Some(false) if made progress, None if popped
    fn resume_top_context(&mut self) -> PyResult<Option<bool>> {
        let idx = self.context_stack.len() - 1;

        // Handle terminal/chance nodes immediately
        if self.context_stack[idx].game_state.is_terminal() {
            let value = self.handle_terminal_node(idx)?;
            self.propagate_value_and_pop(idx, value)?;
            return Ok(Some(false));
        }

        if self.context_stack[idx].game_state.is_chance_node() {
            self.handle_chance_node(idx)?;
            return Ok(Some(false));
        }

        // Handle player nodes with state machine
        match &self.context_stack[idx].stage {
            TraversalStage::PreInference => {
                self.handle_pre_inference(idx)
            }
            TraversalStage::PostInference { .. } => {
                self.handle_post_inference(idx)
            }
            TraversalStage::Finalizing { .. } => {
                self.handle_finalizing(idx)
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

        self.context_stack.push(child_ctx);
        Ok(())
    }

    /// Handle PreInference stage: need network prediction
    fn handle_pre_inference(&mut self, idx: usize) -> PyResult<Option<bool>> {
        let ctx = &self.context_stack[idx];

        // Check if we already have prediction cached
        if let Some(query_id) = ctx.query_id {
            if let Some(strategy) = self.prediction_cache.get(&query_id).cloned() {
                // Prediction available, transition to PostInference
                let num_actions = ctx.legal_actions.len();
                let action_values = vec![0.0; num_actions];

                self.context_stack[idx].stage = TraversalStage::PostInference {
                    strategy,
                    action_values,
                    action_idx: 0,
                };

                return Ok(Some(false));  // Made progress
            }
        }

        // Need to request inference
        let state_encoding = StateEncoder::encode(&ctx.game_state, ctx.update_player);

        if let Some(query_id) = self.query_buffer.add_query(&state_encoding) {
            // Query added, mark context as waiting
            self.context_stack[idx].query_id = Some(query_id);
            Ok(Some(false))  // Made progress, but not blocking yet
        } else {
            // Buffer full, need inference now
            Ok(Some(true))  // Blocking: need inference
        }
    }

    /// Handle PostInference stage: explore actions
    fn handle_post_inference(&mut self, idx: usize) -> PyResult<Option<bool>> {
        let ctx = &self.context_stack[idx];
        let (strategy, action_values, action_idx) = match &ctx.stage {
            TraversalStage::PostInference { strategy, action_values, action_idx } => {
                (strategy.clone(), action_values.clone(), *action_idx)
            }
            _ => unreachable!(),
        };

        if action_idx < ctx.legal_actions.len() {
            // Explore next action
            let action = ctx.legal_actions[action_idx];
            let next_state = ctx.game_state.apply_action(action).expect("Failed to apply action");

            let new_reach_0 = if ctx.current_player == 0 {
                ctx.reach_prob_0 * strategy[action_idx]
            } else {
                ctx.reach_prob_0
            };

            let new_reach_1 = if ctx.current_player == 1 {
                ctx.reach_prob_1 * strategy[action_idx]
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

            // Advance action index
            self.context_stack[idx].stage = TraversalStage::PostInference {
                strategy,
                action_values,
                action_idx: action_idx + 1,
            };

            // Push child
            self.context_stack.push(child_ctx);
            Ok(Some(false))
        } else {
            // All actions explored, transition to Finalizing
            self.context_stack[idx].stage = TraversalStage::Finalizing {
                strategy,
                action_values,
            };
            Ok(Some(false))
        }
    }

    /// Handle Finalizing stage: compute final value and regrets
    fn handle_finalizing(&mut self, idx: usize) -> PyResult<Option<bool>> {
        let (strategy, action_values) = match &self.context_stack[idx].stage {
            TraversalStage::Finalizing { strategy, action_values } => {
                (strategy.clone(), action_values.clone())
            }
            _ => unreachable!(),
        };

        // Compute expected value
        let expected_value: f32 = action_values.iter()
            .zip(strategy.iter())
            .map(|(v, s)| v * s)
            .sum();

        // Compute instant regrets
        let mut regrets = vec![0.0; 4];
        for i in 0..action_values.len().min(4) {
            regrets[i] = action_values[i] - expected_value;
        }

        // Add to replay buffer
        let ctx = &self.context_stack[idx];
        let state_encoding = StateEncoder::encode(&ctx.game_state, ctx.update_player);
        self.buffer.add(&state_encoding, &regrets);

        // Propagate value up and pop
        self.propagate_value_and_pop(idx, expected_value)?;

        Ok(None)  // Context popped
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

        // Update parent's action_values
        if let TraversalStage::PostInference { ref mut action_values, action_idx, .. } =
            self.context_stack[parent_idx].stage
        {
            if action_idx > 0 {
                action_values[action_idx - 1] = value;
            }
        }

        self.context_stack.pop();
        Ok(())
    }
}
