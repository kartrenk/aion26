"""Deep CFR implementation with neural network function approximation.

This module implements Deep Counterfactual Regret Minimization (Deep CFR),
which uses neural networks to approximate cumulative regrets instead of
tabular storage. This allows scaling to larger games.

Key components:
- Advantage network: Approximates cumulative regrets R(I, a)
- Target network: Provides stable bootstrap targets
- Reservoir buffer: Stores (state, regret) experiences
- Bootstrap loss: Combines instant regrets with target predictions

Reference:
- Brown et al. (2019): "Deep Counterfactual Regret Minimization"
- Aion-26 Technical Report: PDCFR+ with dynamic discounting
"""

from typing import Protocol, Optional
from collections import defaultdict
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from aion26.deep_cfr.networks import DeepCFRNetwork, KuhnEncoder, ValueNetwork
from aion26.memory.reservoir import ReservoirBuffer
from aion26.cfr.regret_matching import regret_matching, sample_action
from aion26.learner.discounting import (
    DiscountScheduler,
    PDCFRScheduler,
    LinearScheduler,
    DDCFRStrategyScheduler,
)


class GameState(Protocol):
    """Protocol for game states (duck typing)."""

    def apply_action(self, action: int) -> "GameState": ...
    def legal_actions(self) -> list[int]: ...
    def is_terminal(self) -> bool: ...
    def returns(self) -> tuple[float, float]: ...
    def current_player(self) -> int: ...
    def information_state_string(self) -> str: ...
    def is_chance_node(self) -> bool: ...
    def chance_outcomes(self) -> list[tuple[int, float]]: ...


class DeepCFRTrainer:
    """Deep CFR solver using neural network function approximation.

    This trainer implements the Deep CFR algorithm with bootstrap targets
    for improved sample efficiency. It can be extended to PDCFR+ by adding
    dynamic discounting policies.

    Attributes:
        advantage_net: Neural network for predicting cumulative regrets
        target_net: Target network for stable bootstrap estimates
        encoder: State encoder (e.g., KuhnEncoder)
        buffer: Reservoir buffer for experience replay
        optimizer: Optimizer for advantage network
        iteration: Current iteration count
        discount: Bootstrap discount factor (0.0 = instant regrets only)
        polyak: Polyak averaging coefficient for target network updates
        batch_size: Mini-batch size for training
        train_every: Train networks every N iterations
        target_update_every: Update target network every N iterations
    """

    # Target normalization constant: Max utility (chips) winnable in River Hold'em
    # Used to scale targets to [-1, 1] range for stable neural network training
    MAX_UTILITY = 500.0

    def __init__(
        self,
        initial_state: GameState,
        encoder: KuhnEncoder,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_hidden_layers: int = 3,
        buffer_capacity: int = 10000,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        discount: float = 0.0,
        polyak: float = 0.01,
        train_every: int = 1,
        target_update_every: int = 10,
        seed: int = 42,
        device: str = "cpu",
        regret_scheduler: Optional[DiscountScheduler] = None,
        strategy_scheduler: Optional[DiscountScheduler] = None,
    ):
        """Initialize the Deep CFR trainer.

        Args:
            initial_state: Initial game state
            encoder: State encoder (e.g., KuhnEncoder)
            input_size: Input feature dimension
            output_size: Number of actions
            hidden_size: Hidden layer size for networks
            num_hidden_layers: Number of hidden layers
            buffer_capacity: Reservoir buffer capacity
            learning_rate: Learning rate for optimizer
            batch_size: Mini-batch size for training
            discount: DEPRECATED - use regret_scheduler instead
                     Bootstrap discount factor (0.0 to 1.0)
                     0.0 = instant regrets only (vanilla Deep CFR)
                     >0.0 = bootstrap with target network
            polyak: Polyak averaging coefficient for target updates (0.0 to 1.0)
                   Smaller values = smoother updates (e.g., 0.01)
            train_every: Train networks every N iterations
            target_update_every: Update target network every N iterations
            seed: Random seed for reproducibility
            device: Device for torch tensors ("cpu" or "cuda")
            regret_scheduler: Dynamic discounting for regret updates (PDCFR+)
                             If None, uses PDCFRScheduler(alpha=2.0, beta=0.5)
                             Set to static discount value if discount > 0
            strategy_scheduler: Dynamic weighting for strategy accumulation
                               If None, uses LinearScheduler()
        """
        self.initial_state = initial_state
        self.encoder = encoder
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.iteration = 0

        # Hyperparameters
        self.batch_size = batch_size
        self.discount = discount  # Keep for backward compatibility
        self.polyak = polyak
        self.train_every = train_every
        self.target_update_every = target_update_every
        self.num_actions = output_size

        # Dynamic discounting schedulers (PDCFR+)
        if regret_scheduler is None:
            # Default: PDCFR+ with α=2.0 (quadratic), β=0.5 (sqrt)
            self.regret_scheduler = PDCFRScheduler(alpha=2.0, beta=0.5)
        else:
            self.regret_scheduler = regret_scheduler

        if strategy_scheduler is None:
            # Default: DDCFR with γ=2.0 (quadratic weighting) for strategy accumulation
            # This is the full DDCFR specification: t^γ weighting
            self.strategy_scheduler = DDCFRStrategyScheduler(gamma=2.0)
        else:
            self.strategy_scheduler = strategy_scheduler

        # Networks
        self.advantage_net = DeepCFRNetwork(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            zero_init_output=True  # Critical for PDCFR+ uniform exploration
        ).to(self.device)

        self.target_net = DeepCFRNetwork(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            zero_init_output=True
        ).to(self.device)

        # Initialize target network with hard copy
        self.target_net.copy_weights_from(self.advantage_net, polyak=1.0)
        self.target_net.eval()  # Target network always in eval mode

        # Value network for Variance Reduction (VR-MCCFR)
        self.value_net = ValueNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
        ).to(self.device)

        # Optimizers
        self.optimizer = Adam(self.advantage_net.parameters(), lr=learning_rate)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=learning_rate)

        # Reservoir buffers (tensor-based, O(1) sampling)
        self.buffer = ReservoirBuffer(
            capacity=buffer_capacity,
            input_shape=(input_size,),
            output_size=output_size,
            device=device
        )

        # Value buffer stores (state, actual_return) for baseline training
        self.value_buffer = ReservoirBuffer(
            capacity=buffer_capacity,
            input_shape=(input_size,),
            output_size=1,  # Single value output
            device=device
        )

        # Average strategy tracking (tabular, for Nash convergence verification)
        # This is a hybrid approach: neural regrets + tabular average strategy
        self.strategy_sum: dict[str, npt.NDArray[np.float64]] = {}

        # Training metrics
        self.total_loss = 0.0
        self.num_train_steps = 0

    def get_predicted_regrets(
        self,
        state: GameState,
        player: Optional[int] = None,
        use_target: bool = False
    ) -> torch.Tensor:
        """Get neural network predictions for regrets at a state.

        Args:
            state: Game state
            player: Player perspective (if None, uses current player)
            use_target: If True, use target network instead of advantage network

        Returns:
            Predicted regrets tensor of shape (num_actions,) in CHIP UNITS
        """
        if player is None:
            player = state.current_player()

        # Encode state
        features = self.encoder.encode(state, player)
        features = features.unsqueeze(0).to(self.device)  # Add batch dimension

        # Get predictions (normalized values in [-1, 1])
        network = self.target_net if use_target else self.advantage_net
        with torch.no_grad():
            regrets_normalized = network(features)

        # CRITICAL: Un-scale predictions back to chip units
        # Network predicts normalized values, but regret matching needs chip scale
        regrets = regrets_normalized * self.MAX_UTILITY

        return regrets.squeeze(0)  # Remove batch dimension

    def get_strategy(
        self,
        state: GameState,
        player: Optional[int] = None
    ) -> npt.NDArray[np.float64]:
        """Get current strategy using regret matching on network predictions.

        Args:
            state: Game state
            player: Player perspective (if None, uses current player)

        Returns:
            Strategy (probability distribution over actions)
        """
        # Get predicted regrets from advantage network
        predicted_regrets = self.get_predicted_regrets(state, player, use_target=False)

        # Convert to numpy and apply regret matching
        regrets_np = predicted_regrets.cpu().numpy()
        strategy = regret_matching(regrets_np)

        return strategy

    def traverse(
        self,
        state: GameState,
        update_player: int,
        reach_prob_0: float,
        reach_prob_1: float,
    ) -> float:
        """Recursively traverse the game tree using External Sampling MCCFR.

        External Sampling Monte Carlo CFR samples chance outcomes and opponent
        actions, making traversal linear in game depth instead of exponential
        in game size. This is critical for scaling to large games like Texas Hold'em.

        Key changes from vanilla CFR:
        - Chance nodes: SAMPLE one outcome (not iterate all)
        - Opponent nodes: SAMPLE one action (already implemented below)
        - Update player nodes: Iterate ALL actions (unchanged - needed for regrets)

        Args:
            state: Current game state
            update_player: The player whose regrets we're updating (0 or 1)
            reach_prob_0: Probability that player 0 reaches this state
            reach_prob_1: Probability that player 1 reaches this state

        Returns:
            Expected value for the update_player at this state
        """
        # Terminal node: return payoff
        if state.is_terminal():
            returns = state.returns()
            return returns[update_player]

        # Chance node: SAMPLE one outcome (External Sampling MCCFR)
        if state.is_chance_node():
            # Sample a single outcome based on chance probabilities
            outcomes = state.chance_outcomes()
            actions, probabilities = zip(*outcomes)

            # Sample one action according to the chance distribution
            sampled_action = self.rng.choice(actions, p=probabilities)
            next_state = state.apply_action(sampled_action)

            # Recurse on the sampled outcome only (no weighting - sampling IS the weighting)
            return self.traverse(next_state, update_player, reach_prob_0, reach_prob_1)

        # Player node
        current_player = state.current_player()
        legal_actions = state.legal_actions()
        num_legal = len(legal_actions)

        # Get current strategy from neural network
        strategy_full = self.get_strategy(state, current_player)
        # Slice to only legal actions (network may output more actions than currently legal)
        strategy = strategy_full[:num_legal]

        # If this is the player we're updating, compute counterfactual values
        if current_player == update_player:
            # Variance Reduction: Compute baseline from value network
            # This baseline estimate reduces variance in regret updates
            state_encoding = self.encoder.encode(state, current_player)
            with torch.no_grad():
                baseline_tensor = self.value_net(state_encoding.unsqueeze(0).to(self.device))
                baseline = baseline_tensor.item()  # Scalar value

            # Compute value for each action
            action_values = np.zeros(num_legal, dtype=np.float64)
            for i, action in enumerate(legal_actions):
                next_state = state.apply_action(action)

                # Update reach probabilities when traversing our own actions
                if current_player == 0:
                    action_values[i] = self.traverse(
                        next_state,
                        update_player,
                        reach_prob_0 * strategy[i],
                        reach_prob_1,
                    )
                else:
                    action_values[i] = self.traverse(
                        next_state,
                        update_player,
                        reach_prob_0,
                        reach_prob_1 * strategy[i],
                    )

            # Expected value of this information state
            node_value = np.dot(strategy, action_values)

            # VR-MCCFR: Compute instant counterfactual regrets with baseline
            # regret[a] = (Q(s,a) - baseline) - (V(s) - baseline) = Q(s,a) - V(s)
            # The baseline cancels in the regret formula, but reduces variance
            # when used with importance sampling weights
            instant_regrets = action_values - node_value

            # Store (state, actual_return) in value buffer for baseline training
            # The value network learns to predict node_value to minimize MSE
            return_target = torch.tensor([node_value], dtype=torch.float32)
            self.value_buffer.add(state_encoding, return_target)

            # Weight regrets by opponent reach probability (CFR weighting)
            opponent_reach = reach_prob_1 if current_player == 0 else reach_prob_0

            # Weighted instant regrets
            weighted_regrets = opponent_reach * instant_regrets

            # Bootstrap target: combine instant regret with target network prediction
            # PDCFR+ uses dynamic discounting with separate weights for positive/negative regrets
            # y(I,a) = r_instant(I,a) + w_t(sign) × R_target(I,a)
            target_regrets = self.get_predicted_regrets(
                state,
                current_player,
                use_target=True
            )
            # Slice to only legal actions (matches instant_regrets size)
            target_regrets_np = target_regrets.cpu().numpy()[:num_legal]

            # Get dynamic discount weights for this iteration
            # Use different exponents for positive vs negative target regrets
            w_positive = self.regret_scheduler.get_weight(max(1, self.iteration), "positive")
            w_negative = self.regret_scheduler.get_weight(max(1, self.iteration), "negative")

            # Create discount vector: w_pos where target > 0, w_neg otherwise
            discount_vector = np.where(
                target_regrets_np > 0,
                w_positive,
                w_negative
            )

            # Bootstrap targets with dynamic discounting
            bootstrap_targets = weighted_regrets + discount_vector * target_regrets_np

            # Store experience in reservoir buffer
            # IMPORTANT: Pad bootstrap_targets to full action space size
            # Network outputs all actions, so targets must match this size
            num_actions = len(strategy_full)  # Total actions (from network output size)
            bootstrap_targets_padded = np.zeros(num_actions, dtype=np.float32)
            bootstrap_targets_padded[:num_legal] = bootstrap_targets

            state_encoding = self.encoder.encode(state, current_player)
            target_tensor = torch.from_numpy(bootstrap_targets_padded).float()
            self.buffer.add(state_encoding, target_tensor)

            # Update average strategy (weighted by reach probability and iteration weight)
            # PDCFR+ uses dynamic weighting: recent iterations count more
            # This allows us to extract Nash equilibrium strategy for verification
            # IMPORTANT: Only accumulate after warm-up period to avoid
            # polluting strategy_sum with random strategies from untrained network
            # Changed to 10 iterations (from 500) for faster debugging
            # Production: Use 500+ for full-scale runs
            if self.iteration > 10:
                # DEBUG: Print once when strategy accumulation starts
                if not hasattr(self, '_logged_accumulation'):
                    print(f"DEBUG: Strategy accumulation STARTED at iter {self.iteration}")
                    self._logged_accumulation = True

                own_reach = reach_prob_0 if current_player == 0 else reach_prob_1

                # Get strategy weight for this iteration (typically linear or PDCFR)
                strategy_weight = self.strategy_scheduler.get_weight(max(1, self.iteration))

                info_state = state.information_state_string()

                if info_state not in self.strategy_sum:
                    self.strategy_sum[info_state] = np.zeros(self.num_actions, dtype=np.float64)

                # Accumulate with dynamic weighting (only for legal actions)
                self.strategy_sum[info_state][:num_legal] += own_reach * strategy_weight * strategy

            return node_value

        else:
            # Opponent's node: sample according to their strategy
            action_idx = sample_action(strategy[:num_legal], self.rng)
            action = legal_actions[action_idx]
            next_state = state.apply_action(action)

            return self.traverse(
                next_state,
                update_player,
                reach_prob_0,
                reach_prob_1,
            )

    def train_network(self) -> float:
        """Train the advantage network on buffered experiences.

        Uses mini-batch gradient descent with MSE loss between network
        predictions and bootstrap targets.

        Dynamic Batch Sizing:
        - Scales batch size with buffer fill to maintain healthy coverage (3-10%)
        - Early (buffer <30%): batch=128 (prevents overfitting on sparse data)
        - Mid (buffer 30-70%): batch=512 (balanced coverage)
        - Late (buffer >70%): batch=1024 (matches original successful 12.8% ratio)

        Returns:
            Training loss (MSE)
        """
        # Dynamic batch sizing based on buffer fill percentage
        fill_pct = self.buffer.fill_percentage / 100.0  # Convert to 0-1 range
        if fill_pct < 0.3:
            effective_batch = 128
        elif fill_pct < 0.7:
            effective_batch = 512
        else:
            effective_batch = 1024

        # Train if we have enough samples for a batch
        if len(self.buffer) < effective_batch:
            return 0.0

        # Sample mini-batch from buffer
        states, targets = self.buffer.sample(effective_batch)

        # Move to device
        states = states.to(self.device)
        targets = targets.to(self.device)

        # CRITICAL: Normalize targets to [-1, 1] range
        # Raw targets are chip values (range -500 to +500)
        # Network predicts normalized values for stable gradients
        targets = targets / self.MAX_UTILITY

        # Forward pass
        predictions = self.advantage_net(states)

        # Compute loss (MSE)
        loss = F.mse_loss(predictions, targets)

        # DEBUG logging disabled for production (reduces I/O overhead)
        # Uncomment for debugging scaling issues:
        # if self.iteration % 1000 == 0:
        #     print(f"DEBUG STATS (Iter {self.iteration}):")
        #     print(f"  Targets: Mean={targets.mean().item():.2f}, Std={targets.std().item():.2f}")
        #     print(f"  Predictions: Mean={predictions.mean().item():.2f}, Std={predictions.std().item():.2f}")

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent divergence
        # Critical for River Hold'em with large state space
        torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Track metrics
        self.total_loss += loss.item()
        self.num_train_steps += 1

        return loss.item()

    def train_value_network(self) -> float:
        """Train the value network on buffered state-return pairs.

        The value network learns to predict state values V(s) which are used
        as baselines for variance reduction in VR-MCCFR.

        Uses same dynamic batch sizing as advantage network.

        Returns:
            Training loss (MSE)
        """
        # Dynamic batch sizing (same as advantage network)
        fill_pct = self.value_buffer.fill_percentage / 100.0
        if fill_pct < 0.3:
            effective_batch = 128
        elif fill_pct < 0.7:
            effective_batch = 512
        else:
            effective_batch = 1024

        # Train if we have enough samples for a batch
        if len(self.value_buffer) < effective_batch:
            return 0.0

        # Sample mini-batch from value buffer
        states, returns = self.value_buffer.sample(effective_batch)

        # Move to device
        states = states.to(self.device)
        returns = returns.to(self.device)

        # Forward pass
        predictions = self.value_net(states)

        # Compute loss (MSE between predicted and actual returns)
        loss = F.mse_loss(predictions, returns)

        # Backward pass
        self.value_optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent divergence
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)

        self.value_optimizer.step()

        return loss.item()

    def update_target_network(self) -> None:
        """Update target network using Polyak averaging.

        Soft update: target = polyak × advantage + (1 - polyak) × target
        """
        self.target_net.copy_weights_from(self.advantage_net, polyak=self.polyak)

    def run_iteration(self) -> dict[str, float]:
        """Run one iteration of Deep CFR.

        This includes:
        1. CFR traversal for both players (experience collection)
        2. Network training (if iteration % train_every == 0)
        3. Target network update (if iteration % target_update_every == 0)

        Returns:
            Dictionary of metrics (loss, buffer_size, etc.)
        """
        self.iteration += 1

        # Traverse for player 0
        self.traverse(
            self.initial_state,
            update_player=0,
            reach_prob_0=1.0,
            reach_prob_1=1.0
        )

        # Traverse for player 1
        self.traverse(
            self.initial_state,
            update_player=1,
            reach_prob_0=1.0,
            reach_prob_1=1.0
        )

        # Training metrics
        metrics = {
            "iteration": self.iteration,
            "buffer_size": len(self.buffer),
            "buffer_fill_pct": self.buffer.fill_percentage,
            "loss": 0.0,
            "value_loss": 0.0
        }

        # Train networks
        if self.iteration % self.train_every == 0:
            # Train advantage network (regrets)
            loss = self.train_network()
            metrics["loss"] = loss

            # Train value network (baselines for VR-MCCFR)
            value_loss = self.train_value_network()
            metrics["value_loss"] = value_loss

        # Update target network
        if self.iteration % self.target_update_every == 0:
            self.update_target_network()
            metrics["target_updated"] = True

        return metrics

    def collect_experience(self, num_traversals: int) -> int:
        """Collect experience through game traversals without training.

        This method is used for batch accumulation in Turbo Mode:
        - Run multiple fast traversals (Rust game logic)
        - Add experiences to buffer
        - Do NOT train networks (avoid GPU overhead)

        Args:
            num_traversals: Number of full game traversals to execute

        Returns:
            Number of samples added to buffer
        """
        samples_added = 0

        for _ in range(num_traversals):
            self.iteration += 1

            # Traverse for player 0
            self.traverse(
                self.initial_state,
                update_player=0,
                reach_prob_0=1.0,
                reach_prob_1=1.0
            )
            samples_added += 1

            # Traverse for player 1
            self.traverse(
                self.initial_state,
                update_player=1,
                reach_prob_0=1.0,
                reach_prob_1=1.0
            )
            samples_added += 1

        return samples_added

    def train_step(self) -> dict[str, float]:
        """Execute a single training step on buffered experiences.

        This method is used for batch accumulation in Turbo Mode:
        - Sample batch from buffer
        - Train advantage and value networks
        - Update target network if needed

        Returns:
            Dictionary of training metrics (loss, buffer stats, etc.)
        """
        metrics = {
            "iteration": self.iteration,
            "buffer_size": len(self.buffer),
            "buffer_fill_pct": self.buffer.fill_percentage,
            "loss": 0.0,
            "value_loss": 0.0
        }

        # Train advantage network (regrets)
        if len(self.buffer) >= self.batch_size:
            loss = self.train_network()
            metrics["loss"] = loss

            # Train value network (baselines for VR-MCCFR)
            value_loss = self.train_value_network()
            metrics["value_loss"] = value_loss

        # Update target network (periodic)
        if self.iteration % self.target_update_every == 0:
            self.update_target_network()
            metrics["target_updated"] = True

        return metrics

    def get_average_loss(self) -> float:
        """Get average training loss across all training steps.

        Returns:
            Average loss (0.0 if no training steps)
        """
        if self.num_train_steps == 0:
            return 0.0
        return self.total_loss / self.num_train_steps

    def get_average_strategy(self, info_state: str) -> npt.NDArray[np.float64]:
        """Get average strategy for an information state.

        The average strategy converges to Nash equilibrium in two-player zero-sum games.

        Args:
            info_state: Information state string

        Returns:
            Average strategy (probability distribution over actions)
        """
        if info_state not in self.strategy_sum:
            # If never visited, return uniform
            return np.ones(self.num_actions, dtype=np.float64) / self.num_actions

        strat_sum = self.strategy_sum[info_state]
        total = strat_sum.sum()

        if total <= 0.0:
            # If sum is zero, return uniform
            return np.ones(self.num_actions, dtype=np.float64) / self.num_actions

        return strat_sum / total

    def get_all_average_strategies(self) -> dict[str, npt.NDArray[np.float64]]:
        """Get average strategies for all visited information states.

        Returns:
            Dictionary mapping info_state -> average strategy
        """
        return {
            info_state: self.get_average_strategy(info_state)
            for info_state in self.strategy_sum.keys()
        }
