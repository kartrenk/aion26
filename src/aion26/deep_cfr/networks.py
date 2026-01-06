"""Neural network components for Deep CFR.

This module provides:
- CardEmbedding: One-hot encoding for card ranks
- KuhnEncoder: State encoder for Kuhn Poker
- DeepCFRNetwork: MLP for regret/strategy approximation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from aion26.games.kuhn import KuhnPoker, JACK, QUEEN, KING
from aion26.games.leduc import LeducPoker, Card


class CardEmbedding:
    """Helper to convert card ranks (J, Q, K) into one-hot vectors.

    This is a simple utility class that maps card integers to one-hot vectors:
    - JACK (0) -> [1, 0, 0]
    - QUEEN (1) -> [0, 1, 0]
    - KING (2) -> [0, 0, 1]
    """

    @staticmethod
    def encode(card: int) -> np.ndarray:
        """Encode a card as a one-hot vector.

        Args:
            card: Integer card rank (0=J, 1=Q, 2=K)

        Returns:
            One-hot numpy array of shape (3,)

        Raises:
            ValueError: If card is not in {0, 1, 2}
        """
        if card not in {JACK, QUEEN, KING}:
            raise ValueError(f"Invalid card: {card}. Must be JACK(0), QUEEN(1), or KING(2)")

        one_hot = np.zeros(3, dtype=np.float32)
        one_hot[card] = 1.0
        return one_hot

    @staticmethod
    def to_tensor(card: int) -> torch.FloatTensor:
        """Encode a card as a one-hot tensor.

        Args:
            card: Integer card rank (0=J, 1=Q, 2=K)

        Returns:
            One-hot torch tensor of shape (3,)
        """
        return torch.from_numpy(CardEmbedding.encode(card))


class KuhnEncoder:
    """Encoder that converts a KuhnPoker state into a feature tensor.

    Features extracted:
    1. My Card (one-hot, 3 dims): Which card the current player holds
    2. Betting History (binary flags, 6 dims): Actions taken so far
       - For each of up to 3 actions: [is_check, is_bet]
    3. Pot Size (normalized, 1 dim): Current pot divided by max pot (5 chips)

    Total: 10 dimensions

    Example:
        state = KuhnPoker(cards=(JACK, QUEEN), history="cb")
        encoder = KuhnEncoder()
        features = encoder.encode(state, player=0)
        # features will be a tensor of shape (10,)
    """

    def __init__(self, max_pot: float = 5.0):
        """Initialize encoder.

        Args:
            max_pot: Maximum possible pot size for normalization.
                    In Kuhn Poker, max pot is 5 (2 ante + 1 + 1 + 1)
        """
        self.max_pot = max_pot

    def encode(self, state: KuhnPoker, player: Optional[int] = None) -> torch.FloatTensor:
        """Encode a Kuhn Poker state into a feature tensor.

        Args:
            state: The KuhnPoker game state
            player: Which player's perspective to encode from.
                   If None, uses state.current_player()

        Returns:
            Feature tensor of shape (10,) containing:
            - Card one-hot (3 dims)
            - Betting history (6 dims)
            - Normalized pot size (1 dim)

        Raises:
            ValueError: If state is terminal or chance node
        """
        if player is None:
            player = state.current_player()

        if player == -1:
            raise ValueError("Cannot encode terminal or chance node state")

        # 1. Card encoding (3 dims) - one-hot
        card = state.cards[player]
        if card is None:
            raise ValueError("Card not yet dealt")
        card_features = CardEmbedding.to_tensor(card)

        # 2. Betting history (6 dims) - binary flags
        # Encode up to 3 actions, each as [is_check, is_bet]
        history_features = torch.zeros(6, dtype=torch.float32)
        for i, action_char in enumerate(state.history):
            if i >= 3:  # Max 3 actions in Kuhn
                break
            if action_char == "c":
                history_features[i * 2] = 1.0  # Check flag
            else:  # "b"
                history_features[i * 2 + 1] = 1.0  # Bet flag

        # 3. Pot size (1 dim) - normalized by max pot
        pot_features = torch.tensor([state.pot / self.max_pot], dtype=torch.float32)

        # Concatenate all features
        return torch.cat([card_features, history_features, pot_features])

    def feature_size(self) -> int:
        """Get the dimensionality of the encoded features.

        Returns:
            10 (3 card + 6 history + 1 pot)
        """
        return 10


class DeepCFRNetwork(nn.Module):
    """MLP for approximating cumulative regrets (Advantage Network).

    Architecture:
    - Input layer: feature_size (e.g., 10 for Kuhn)
    - Hidden layer 1: 64 units + ReLU
    - Hidden layer 2: 64 units + ReLU
    - Hidden layer 3: 64 units + ReLU
    - Output layer: num_actions (e.g., 2 for Kuhn)

    This network predicts cumulative regrets for each action.
    During CFR, we use regret matching on these outputs to get strategies.

    Example:
        encoder = KuhnEncoder()
        network = DeepCFRNetwork(input_size=encoder.feature_size(), output_size=2)

        state = KuhnPoker(cards=(JACK, QUEEN), history="")
        features = encoder.encode(state, player=0)
        regrets = network(features.unsqueeze(0))  # Shape: (1, 2)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_hidden_layers: int = 3,
        zero_init_output: bool = True
    ):
        """Initialize the network.

        Args:
            input_size: Dimensionality of input features
            output_size: Number of actions (output dimensions)
            hidden_size: Number of units in each hidden layer (default: 64)
            num_hidden_layers: Number of hidden layers (default: 3)
            zero_init_output: If True, initialize output layer with near-zero weights
                            (default: True). This ensures near-uniform exploration at
                            start of training, avoiding premature convergence.
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # Build network layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer (no activation - raw regret values)
        output_layer = nn.Linear(hidden_size, output_size)

        # Initialize output layer with near-zero weights for uniform exploration
        # This is critical for PDCFR+ to avoid premature convergence
        if zero_init_output:
            nn.init.normal_(output_layer.weight, mean=0.0, std=0.001)
            nn.init.zeros_(output_layer.bias)

        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
            representing predicted regrets for each action
        """
        return self.network(x)

    def copy_weights_from(self, source_network: "DeepCFRNetwork", polyak: float = 1.0):
        """Copy or blend weights from source network using Polyak averaging.

        This method is essential for Deep PDCFR+ which uses a target network
        for bootstrapped value estimates. The Polyak averaging provides stability
        by slowly updating the target network.

        Args:
            source_network: Network to copy weights from
            polyak: Polyak averaging coefficient (default: 1.0)
                   - polyak=1.0: Hard copy (full replacement)
                     target = source
                   - polyak<1.0: Soft update (exponential moving average)
                     target = polyak * source + (1-polyak) * target

        Raises:
            ValueError: If networks have incompatible architectures
            ValueError: If polyak not in [0, 1]

        Example:
            # Hard copy (target network = source network)
            target_net.copy_weights_from(source_net, polyak=1.0)

            # Soft update (target slowly tracks source)
            target_net.copy_weights_from(source_net, polyak=0.01)
        """
        # Validate polyak coefficient
        if not 0.0 <= polyak <= 1.0:
            raise ValueError(f"Polyak must be in [0, 1], got {polyak}")

        # Validate architecture compatibility
        if (self.input_size != source_network.input_size or
            self.output_size != source_network.output_size or
            self.hidden_size != source_network.hidden_size or
            self.num_hidden_layers != source_network.num_hidden_layers):
            raise ValueError(
                f"Cannot copy weights between incompatible networks:\n"
                f"  Target: {repr(self)}\n"
                f"  Source: {repr(source_network)}"
            )

        # Copy/blend weights using Polyak averaging
        with torch.no_grad():
            for target_param, source_param in zip(
                self.parameters(),
                source_network.parameters()
            ):
                if polyak == 1.0:
                    # Hard copy
                    target_param.data.copy_(source_param.data)
                else:
                    # Soft update: target = polyak * source + (1-polyak) * target
                    target_param.data.mul_(1.0 - polyak)
                    target_param.data.add_(source_param.data, alpha=polyak)

    def __repr__(self) -> str:
        """String representation of the network."""
        return (
            f"DeepCFRNetwork("
            f"input={self.input_size}, "
            f"hidden={self.hidden_size}x{self.num_hidden_layers}, "
            f"output={self.output_size})"
        )


class LeducEncoder:
    """Encoder that converts a Leduc Poker state into a feature tensor.

    Features extracted:
    1. Private Card (one-hot, 6 dims): 3 ranks × 2 suits
    2. Public Card (one-hot, 6 dims): Same as above, zeros if not dealt
    3. Current Round (1 dim): 0=round1, 1=round2
    4. Betting History Round 1 (6 dims): Up to 3 actions × [is_check, is_bet]
    5. Betting History Round 2 (6 dims): Up to 3 actions × [is_check, is_bet]
    6. Pot Size (1 dim): Normalized by max pot (~20 chips)

    Total: 6 + 6 + 1 + 6 + 6 + 1 = 26 dimensions

    Example:
        state = LeducPoker(cards=(Card(0,0), Card(1,0), None), history="")
        encoder = LeducEncoder()
        features = encoder.encode(state, player=0)
        # features will be a tensor of shape (26,)
    """

    def __init__(self, max_pot: float = 20.0):
        """Initialize encoder.

        Args:
            max_pot: Maximum possible pot size for normalization.
                    In Leduc, max pot is ~20 (2 antes + multiple bets)
        """
        self.max_pot = max_pot

    def _encode_card(self, card: Optional[Card]) -> np.ndarray:
        """Encode a Leduc card (rank + suit) as one-hot vector.

        Args:
            card: Card object with rank and suit, or None

        Returns:
            One-hot vector of shape (6,)
            Index = rank * 2 + suit
        """
        if card is None:
            return np.zeros(6, dtype=np.float32)

        # Map to index: J♠=0, J♥=1, Q♠=2, Q♥=3, K♠=4, K♥=5
        index = card.rank * 2 + card.suit
        one_hot = np.zeros(6, dtype=np.float32)
        one_hot[index] = 1.0
        return one_hot

    def _encode_history(self, history: str) -> np.ndarray:
        """Encode betting history into binary flags.

        Args:
            history: Betting history string (e.g., "cb/bb")

        Returns:
            Binary flags of shape (12,)
            First 6 dims: round 1 actions (up to 3 actions × 2 flags)
            Last 6 dims: round 2 actions (up to 3 actions × 2 flags)
        """
        # Split history by round
        if '/' in history:
            round1_history, round2_history = history.split('/', 1)
        else:
            round1_history = history
            round2_history = ""

        features = np.zeros(12, dtype=np.float32)

        # Encode round 1 (first 6 dims)
        for i, action_char in enumerate(round1_history):
            if i >= 3:  # Max 3 actions per round
                break
            if action_char == "c":
                features[i * 2] = 1.0  # Check flag
            elif action_char == "b":
                features[i * 2 + 1] = 1.0  # Bet flag

        # Encode round 2 (last 6 dims)
        for i, action_char in enumerate(round2_history):
            if i >= 3:
                break
            if action_char == "c":
                features[6 + i * 2] = 1.0  # Check flag
            elif action_char == "b":
                features[6 + i * 2 + 1] = 1.0  # Bet flag

        return features

    def encode(self, state: LeducPoker, player: Optional[int] = None) -> torch.FloatTensor:
        """Encode a Leduc Poker state into a feature tensor.

        Args:
            state: The LeducPoker game state
            player: Which player's perspective to encode from.
                   If None, uses state.current_player()

        Returns:
            Feature tensor of shape (26,) containing:
            - Private card one-hot (6 dims)
            - Public card one-hot (6 dims)
            - Current round (1 dim)
            - Betting history round 1 (6 dims)
            - Betting history round 2 (6 dims)
            - Normalized pot size (1 dim)

        Raises:
            ValueError: If state is terminal or chance node
        """
        if player is None:
            player = state.current_player()

        if player == -1:
            raise ValueError("Cannot encode terminal or chance node state")

        # 1. Private card encoding (6 dims)
        private_card = state.cards[player]
        if private_card is None:
            raise ValueError("Private card not yet dealt")
        private_features = self._encode_card(private_card)

        # 2. Public card encoding (6 dims)
        public_card = state.cards[2]
        public_features = self._encode_card(public_card)

        # 3. Current round (1 dim)
        # 0 = round 1, 1 = round 2
        round_feature = np.array([float(state.round - 1)], dtype=np.float32)

        # 4. Betting history (12 dims total: 6 for each round)
        history_features = self._encode_history(state.history)

        # 5. Pot size (1 dim) - normalized
        pot_feature = np.array([state.pot / self.max_pot], dtype=np.float32)

        # Concatenate all features
        all_features = np.concatenate([
            private_features,   # 6 dims
            public_features,    # 6 dims
            round_feature,      # 1 dim
            history_features,   # 12 dims
            pot_feature         # 1 dim
        ])  # Total: 26 dims

        return torch.from_numpy(all_features)

    def feature_size(self) -> int:
        """Get the dimensionality of the encoded features.

        Returns:
            26 (6 private + 6 public + 1 round + 12 history + 1 pot)
        """
        return 26
