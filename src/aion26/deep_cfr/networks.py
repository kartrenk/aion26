"""Neural network components for Deep CFR.

This module provides:
- CardEmbedding: One-hot encoding for card ranks
- KuhnEncoder: State encoder for Kuhn Poker
- LeducEncoder: State encoder for Leduc Poker
- HoldemEncoder: State encoder for Texas Hold'em River (with hand rank features)
- DeepCFRNetwork: MLP for regret/strategy approximation
- ValueNetwork: MLP for state value estimation (Variance Reduction baseline)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from aion26.games.kuhn import KuhnPoker, JACK, QUEEN, KING
from aion26.games.leduc import LeducPoker, Card
from aion26.games.river_holdem import TexasHoldemRiver

# Import treys for hand evaluation
try:
    from treys import Card as TreysCard, Evaluator
    TREYS_AVAILABLE = True
except ImportError:
    TREYS_AVAILABLE = False


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
        self.input_size = 10  # Card (3) + History (6) + Pot (1)

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
        self.input_size = 26  # Private card (6) + Public card (6) + Round (1) + History R1 (6) + History R2 (6) + Pot (1)

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


class ValueNetwork(nn.Module):
    """MLP for estimating state values (Variance Reduction baseline).

    This network is used in VR-MCCFR to provide a baseline for variance reduction.
    It predicts the expected return from a given information state, which is then
    subtracted from action values to reduce the variance of regret estimates.

    Mathematical Foundation:
        Without baseline: regret[a] = utility[a] * weight
        With baseline:    regret[a] = (utility[a] - V(s)) * weight

        Both are unbiased estimators of the true regret, but the baseline version
        has lower variance, leading to faster and more stable convergence.

    Architecture:
    - Same as DeepCFRNetwork but with output_size=1
    - Input layer: feature_size (e.g., 10 for Kuhn, 26 for Leduc)
    - Hidden layers: Configurable (default: 3 × 64 units)
    - Output layer: 1 scalar value (no activation)

    Example:
        encoder = LeducEncoder()
        value_net = ValueNetwork(input_size=26, hidden_size=128, num_hidden_layers=3)

        state = LeducPoker(...)
        features = encoder.encode(state, player=0)
        baseline = value_net(features.unsqueeze(0))  # Shape: (1, 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_hidden_layers: int = 3,
    ):
        """Initialize the value network.

        Args:
            input_size: Dimensionality of input features
            hidden_size: Number of units in each hidden layer (default: 64)
            num_hidden_layers: Number of hidden layers (default: 3)
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # Build network layers (same as DeepCFRNetwork but output_size=1)
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer: single scalar value (expected return)
        # No special initialization needed - standard variance is fine
        output_layer = nn.Linear(hidden_size, 1)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, 1)
            representing predicted state value (expected return)
        """
        return self.network(x)

    def __repr__(self) -> str:
        """String representation of the network."""
        return (
            f"ValueNetwork("
            f"input={self.input_size}, "
            f"hidden={self.hidden_size}x{self.num_hidden_layers}, "
            f"output=1)"
        )


class ResidualBlock(nn.Module):
    """Pre-activation residual block with LayerNorm.

    Structure: LayerNorm → ReLU → Linear → LayerNorm → ReLU → Linear + skip

    Pre-activation design (He et al., 2016) improves gradient flow
    and allows training deeper networks.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.linear1(x)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x + residual


class ResNetDeepCFR(nn.Module):
    """ResNet-based Deep CFR network with Card Embeddings and VR-PDCFR+ support.

    Phase 7 Architecture (500k+ params):
    - Card Embeddings: nn.Embedding(52, 64) for learned card representations
    - Hole Cards: 2 × 64 = 128 dims (concatenated)
    - Board Cards: 5 × 64 → sum pooling → 64 dims (permutation invariant)
    - Context Features: 10 dims (pot odds, stacks, bets)
    - Trunk: 4 residual blocks with LayerNorm + ReLU
    - Dual Heads: Advantage (4 actions) + Value (1 scalar)

    Why this architecture?
    1. Learned Embeddings: The network learns card semantics (e.g., Ace is high,
       flush draws share suit embeddings). More expressive than one-hot.

    2. Permutation Invariance: Summing board embeddings ensures {A♠, K♦, Q♥, J♣, T♠}
       encodes the same regardless of card order. Critical for board texture.

    3. Residual Blocks: Deep networks with skip connections learn hierarchical
       poker concepts (blockers, equity, fold equity, pot odds interactions).

    4. Dual Heads for VR-PDCFR+:
       - Advantage head: Predicts cumulative regrets for each action
       - Value head: Predicts expected value V(s) as variance reduction baseline
       - Loss: || Adv_pred - (Regret_raw - V_baseline) ||²

    Mathematical Foundation (VR-PDCFR+):
        Standard Deep CFR: target = regret_raw
        VR-Deep CFR:       target = regret_raw - V(s)

        The baseline V(s) is trained to minimize the variance of regret estimates.
        Since E[regret] ≈ 0 at equilibrium, centering around V(s) gives lower
        variance targets, enabling faster convergence with fewer samples.

    Example:
        net = ResNetDeepCFR(num_actions=4, card_dim=64, hidden_dim=256, num_blocks=4)
        # Input: batch of card indices and context features
        cards = torch.randint(0, 52, (batch_size, 7))  # 2 hole + 5 board
        context = torch.randn(batch_size, 10)
        advantage, value = net(cards, context)
        # advantage: (batch_size, 4), value: (batch_size, 1)
    """

    def __init__(
        self,
        num_actions: int = 4,
        card_dim: int = 64,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        context_dim: int = 10,
    ):
        """Initialize the ResNet Deep CFR network.

        Args:
            num_actions: Number of actions (default: 4 for fold/call/raise_half/raise_pot)
            card_dim: Card embedding dimension (default: 64)
            hidden_dim: Hidden dimension for residual blocks (default: 256)
            num_blocks: Number of residual blocks (default: 4)
            context_dim: Dimension of context features (default: 10)
        """
        super().__init__()

        self.num_actions = num_actions
        self.card_dim = card_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.context_dim = context_dim

        # Card embeddings: 52 cards → card_dim dimensions
        self.card_embedding = nn.Embedding(52, card_dim)

        # Input projection: hole cards (2×64) + board sum (64) + context (10)
        # Total: 128 + 64 + 10 = 202 → hidden_dim
        input_dim = card_dim * 2 + card_dim + context_dim  # 202
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual trunk: 4 blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        # Output normalization (pre-head)
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Dual heads for VR-PDCFR+
        self.advantage_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Initialize advantage head with near-zero weights for uniform exploration
        nn.init.normal_(self.advantage_head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.advantage_head.bias)

        # Value head uses standard initialization
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self,
        cards: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            cards: Card indices tensor of shape (batch_size, 7)
                   First 2 are hole cards, last 5 are board cards (0-51)
            context: Context features of shape (batch_size, context_dim)
                     Contains pot odds, stacks, bets, etc.

        Returns:
            Tuple of (advantage, value):
            - advantage: Shape (batch_size, num_actions) - cumulative regret estimates
            - value: Shape (batch_size, 1) - state value baseline V(s)
        """
        batch_size = cards.shape[0]

        # Embed cards: (batch, 7, card_dim)
        card_embeds = self.card_embedding(cards)

        # Hole cards: concatenate first 2 embeddings → (batch, 2 * card_dim)
        hole_embeds = card_embeds[:, :2, :].reshape(batch_size, -1)

        # Board cards: sum pool last 5 embeddings → (batch, card_dim)
        # Permutation invariant: order of board cards doesn't matter
        board_embeds = card_embeds[:, 2:, :].sum(dim=1)

        # Concatenate all inputs: (batch, 2*card_dim + card_dim + context_dim)
        x = torch.cat([hole_embeds, board_embeds, context], dim=1)

        # Project to hidden dimension
        x = self.input_proj(x)
        x = torch.relu(x)

        # Residual trunk
        for block in self.blocks:
            x = block(x)

        # Output normalization
        x = self.output_norm(x)
        x = torch.relu(x)

        # Dual heads
        advantage = self.advantage_head(x)
        value = self.value_head(x)

        return advantage, value

    def forward_from_flat(self, flat_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass from flattened 136-dim state vector (for compatibility).

        This method allows using the ResNet with the existing 136-dim encoding
        from HoldemEncoder, extracting card indices and context features.

        Args:
            flat_state: Shape (batch_size, 136) - the standard HoldemEncoder output
                       Layout: [10 rank + 34 hole + 85 board + 7 context]

        Returns:
            Tuple of (advantage, value) as in forward()

        Note: This is less efficient than using raw card indices with forward().
              Use forward() directly when possible.
        """
        batch_size = flat_state.shape[0]

        # Extract card indices from one-hot encoding
        # Hole cards: dims 10-43 (2 cards × 17 bits each)
        # Board cards: dims 44-128 (5 cards × 17 bits each)
        hole_start = 10
        board_start = 44

        cards = torch.zeros(batch_size, 7, dtype=torch.long, device=flat_state.device)

        # Decode hole cards (2 cards)
        for i in range(2):
            offset = hole_start + i * 17
            rank_onehot = flat_state[:, offset:offset+13]
            suit_onehot = flat_state[:, offset+13:offset+17]
            rank = rank_onehot.argmax(dim=1)  # 0-12
            suit = suit_onehot.argmax(dim=1)  # 0-3
            cards[:, i] = suit * 13 + rank  # Convert to 0-51 format

        # Decode board cards (5 cards)
        for i in range(5):
            offset = board_start + i * 17
            rank_onehot = flat_state[:, offset:offset+13]
            suit_onehot = flat_state[:, offset+13:offset+17]
            rank = rank_onehot.argmax(dim=1)
            suit = suit_onehot.argmax(dim=1)
            cards[:, 2+i] = suit * 13 + rank

        # Extract context features (last 7 dims) and pad to context_dim
        context_raw = flat_state[:, -7:]
        if self.context_dim > 7:
            # Pad with zeros if context_dim > 7
            context = torch.zeros(batch_size, self.context_dim, device=flat_state.device)
            context[:, :7] = context_raw
        else:
            context = context_raw[:, :self.context_dim]

        return self.forward(cards, context)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation with parameter count."""
        params = self.count_parameters()
        return (
            f"ResNetDeepCFR("
            f"actions={self.num_actions}, "
            f"card_dim={self.card_dim}, "
            f"hidden={self.hidden_dim}, "
            f"blocks={self.num_blocks}, "
            f"params={params:,})"
        )


class HoldemEncoder:
    """Encoder that converts a Texas Hold'em River state into a feature tensor.

    FIXED: Full one-hot encoding for sharp representation (no normalized floats).

    Features extracted:
    1. Hand Rank Category (one-hot, 10 dims): Helper feature from treys evaluator
       - 0: High Card, 1: One Pair, 2: Two Pair, 3: Three of a Kind
       - 4: Straight, 5: Flush, 6: Full House, 7: Four of a Kind
       - 8: Straight Flush, 9: Royal Flush

    2. Hole Cards (34 dims): Full one-hot encoding for 2 hole cards
       - Each card: 13 rank bits (Deuce→Ace) + 4 suit bits (♠♥♦♣)
       - 2 cards × 17 bits = 34 binary features

    3. Board Cards (85 dims): Full one-hot encoding for 5 board cards
       - Each card: 13 rank bits + 4 suit bits = 17 bits
       - 5 cards × 17 bits = 85 binary features

    4. Betting Context (7 dims):
       - Pot size (normalized), Player 0/1 stacks (normalized)
       - Current bet (normalized), Player 0/1 invested (normalized)
       - Pot odds (call_amount / (pot + call_amount))

    Total: 10 + 34 + 85 + 7 = 136 dimensions

    Why one-hot? Sharp binary features allow the network to learn discrete
    card identities (e.g., "if Ace present, then bet"). Normalized floats
    (0.917 vs 1.0) are inefficient for learning poker discontinuities.

    Example:
        state = TexasHoldemRiver(...)
        encoder = HoldemEncoder()
        features = encoder.encode(state)
        # features will be a tensor of shape (136,)
    """

    def __init__(self, max_pot: float = 500.0, max_stack: float = 200.0):
        """Initialize encoder.

        Args:
            max_pot: Maximum pot size for normalization (default: 500)
            max_stack: Maximum stack size for normalization (default: 200)
        """
        if not TREYS_AVAILABLE:
            raise ImportError("treys library is required for HoldemEncoder. Install with: pip install treys")

        self.max_pot = max_pot
        self.max_stack = max_stack
        self.input_size = 136  # Hand rank (10) + Hole cards (34) + Board (85) + Context (7)
        self.evaluator = Evaluator()

        # Try to import Rust evaluator for faster evaluation
        try:
            import aion26_rust
            self.rust_evaluator = aion26_rust
        except ImportError:
            self.rust_evaluator = None

    def _get_hand_rank_category(self, hand: list[int], board: list[int]) -> int:
        """Get hand rank category (0-9) using evaluator.

        Args:
            hand: 2 hole cards (treys or Rust 0-51 format)
            board: 5 board cards (treys or Rust 0-51 format)

        Returns:
            Integer 0-9 representing hand rank category
        """
        # Detect card format: Rust cards are 0-51, treys cards are large bit flags
        is_rust_format = all(0 <= c <= 51 for c in hand + board)

        if is_rust_format and self.rust_evaluator is not None:
            # Use Rust evaluator (expects 7 cards in 0-51 format)
            seven_cards = hand + board
            rank = self.rust_evaluator.evaluate_7_cards(seven_cards)
            category = self.rust_evaluator.get_category(rank)
            return category
        else:
            # Use treys evaluator (expects separate hand and board)
            rank = self.evaluator.evaluate(board, hand)

            # Convert to category (0-9)
            # Treys rank ranges: 1-7462 (lower is better)
            if rank == 1:
                return 9  # Royal Flush
            elif rank <= 10:
                return 8  # Straight Flush
            elif rank <= 166:
                return 7  # Four of a Kind
            elif rank <= 322:
                return 6  # Full House
            elif rank <= 1599:
                return 5  # Flush
            elif rank <= 1609:
                return 4  # Straight
            elif rank <= 2467:
                return 3  # Three of a Kind
            elif rank <= 3325:
                return 2  # Two Pair
            elif rank <= 6185:
                return 1  # One Pair
            else:
                return 0  # High Card

    def _encode_card(self, card: int) -> np.ndarray:
        """Encode a single card as one-hot rank + suit.

        Args:
            card: Card in treys or Rust 0-51 format

        Returns:
            Array of shape (17,) with binary features:
            - First 13 bits: rank one-hot (Deuce=0, ..., Ace=12)
            - Last 4 bits: suit one-hot (Spades=0, Hearts=1, Diamonds=2, Clubs=3)

        Example:
            Ace of Spades: [0,0,0,0,0,0,0,0,0,0,0,0,1, 1,0,0,0]
            King of Hearts: [0,0,0,0,0,0,0,0,0,0,0,1,0, 0,1,0,0]
        """
        # Detect card format
        if 0 <= card <= 51:
            # Rust format: 0-51 encoding
            rank = card % 13  # 0-12 (Deuce to Ace)
            suit = card // 13  # 0-3
        else:
            # Treys format: bit flags
            rank = TreysCard.get_rank_int(card)  # 0-12 (Deuce to Ace)
            suit_bits = TreysCard.get_suit_int(card)  # 1, 2, 4, 8 (bit flags)
            # Convert bit flag to index: 1→0, 2→1, 4→2, 8→3
            suit = suit_bits.bit_length() - 1

        # Create one-hot vectors
        rank_one_hot = np.zeros(13, dtype=np.float32)
        rank_one_hot[rank] = 1.0

        suit_one_hot = np.zeros(4, dtype=np.float32)
        suit_one_hot[suit] = 1.0

        return np.concatenate([rank_one_hot, suit_one_hot])

    def encode(self, state: TexasHoldemRiver, player: Optional[int] = None) -> torch.FloatTensor:
        """Encode a Texas Hold'em River state into a feature tensor.

        Args:
            state: The TexasHoldemRiver game state
            player: Which player's perspective to encode from.
                   If None, uses state.current_player()

        Returns:
            Feature tensor of shape (136,) containing:
            - Hand rank category one-hot (10 dims)
            - Hole cards one-hot (34 dims: 2 cards × 17 bits each)
            - Board cards one-hot (85 dims: 5 cards × 17 bits each)
            - Betting context (7 dims)

        Raises:
            ValueError: If state is terminal, chance node, or cards not dealt
        """
        if player is None:
            player = state.current_player()

        if player == -1:
            raise ValueError("Cannot encode terminal or chance node state")

        if not state.is_dealt:
            raise ValueError("Cannot encode state before cards are dealt")

        # 1. Hand rank category (one-hot, 10 dims)
        hand = state.hands[player]
        board = state.board
        rank_category = self._get_hand_rank_category(hand, board)

        rank_features = np.zeros(10, dtype=np.float32)
        rank_features[rank_category] = 1.0

        # 2. Hole cards features (34 dims: 2 cards × 17 bits each)
        hole_features = np.concatenate([
            self._encode_card(hand[0]),  # 17 dims
            self._encode_card(hand[1])   # 17 dims
        ])

        # 3. Board cards features (85 dims: 5 cards × 17 bits each)
        board_features = np.concatenate([
            self._encode_card(board[i]) for i in range(5)  # 5 × 17 dims
        ])

        # 4. Betting context (7 dims)
        current_invested = state.player_0_invested if player == 0 else state.player_1_invested
        opponent_invested = state.player_1_invested if player == 0 else state.player_0_invested

        # Calculate pot odds
        call_amount = max(0, state.current_bet - current_invested)
        pot_after_call = state.pot + call_amount
        pot_odds = call_amount / pot_after_call if pot_after_call > 0 else 0.0

        context_features = np.array([
            state.pot / self.max_pot,
            state.stacks[0] / self.max_stack,
            state.stacks[1] / self.max_stack,
            state.current_bet / self.max_stack,
            state.player_0_invested / self.max_stack,
            state.player_1_invested / self.max_stack,
            pot_odds
        ], dtype=np.float32)

        # Concatenate all features
        all_features = np.concatenate([
            rank_features,    # 10 dims
            hole_features,    # 34 dims
            board_features,   # 85 dims
            context_features  # 7 dims
        ])  # Total: 136 dims

        return torch.from_numpy(all_features)

    def feature_size(self) -> int:
        """Get the dimensionality of the encoded features.

        Returns:
            136 (10 rank + 34 hole + 85 board + 7 context)
        """
        return 136
