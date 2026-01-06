"""Configuration system for reproducible Deep PDCFR+ experiments.

Adapted from poker_solver-main JSON config pattern for Deep RL workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Any
import yaml


@dataclass
class GameConfig:
    """Game selection and parameters."""

    name: Literal["kuhn", "leduc", "river_holdem"] = "leduc"
    # Future: Add game-specific parameters here
    # e.g., leduc_suits: int = 2, leduc_ranks: int = 3
    # river_holdem: pot size, stack size, etc.


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""

    iterations: int = 2000  # Increased for demo-friendly behavior
    batch_size: int = 128
    buffer_capacity: int = 1000  # Reduced for quick GUI demos (fills in ~400 iterations)
    eval_every: int = 100  # Evaluate NashConv every N iterations
    log_every: int = 10    # Log metrics every N iterations


@dataclass
class ModelConfig:
    """Neural network architecture."""

    hidden_size: int = 128
    num_hidden_layers: int = 4
    learning_rate: float = 0.001
    # Optimizer settings
    optimizer: Literal["adam", "sgd"] = "adam"
    weight_decay: float = 0.0


@dataclass
class AlgorithmConfig:
    """CFR algorithm variant configuration."""

    use_vr: bool = True  # Variance Reduction with Value Network
    scheduler_type: Literal["uniform", "linear", "pdcfr", "ddcfr"] = "ddcfr"
    # PDCFR/DDCFR discounting parameters
    alpha: float = 1.5   # Positive regret discount
    beta: float = 0.0    # Negative regret discount
    gamma: float = 2.0   # Strategy discount (for DDCFR)


@dataclass
class AionConfig:
    """Complete configuration for a Deep PDCFR+ experiment.

    Enables reproducible experiments with YAML serialization.

    Example:
        >>> config = AionConfig(
        ...     game=GameConfig(name="leduc"),
        ...     algorithm=AlgorithmConfig(use_vr=True, scheduler_type="ddcfr")
        ... )
        >>> config.to_yaml("experiment.yaml")
        >>> loaded = AionConfig.from_yaml("experiment.yaml")
    """

    game: GameConfig = field(default_factory=GameConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)

    # Experiment metadata
    name: str = "default_experiment"
    seed: int = 42
    device: Literal["cpu", "cuda"] = "cpu"

    @classmethod
    def from_yaml(cls, path: str | Path) -> AionConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            AionConfig instance.

        Raises:
            FileNotFoundError: If path does not exist.
            yaml.YAMLError: If file is not valid YAML.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Reconstruct nested dataclasses
        game = GameConfig(**data.get("game", {}))
        training = TrainingConfig(**data.get("training", {}))
        model = ModelConfig(**data.get("model", {}))
        algorithm = AlgorithmConfig(**data.get("algorithm", {}))

        # Top-level fields
        metadata = {k: v for k, v in data.items() if k not in ["game", "training", "model", "algorithm"]}

        return cls(
            game=game,
            training=training,
            model=model,
            algorithm=algorithm,
            **metadata
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Destination path for YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to nested dict
        data = {
            "name": self.name,
            "seed": self.seed,
            "device": self.device,
            "game": asdict(self.game),
            "training": asdict(self.training),
            "model": asdict(self.model),
            "algorithm": asdict(self.algorithm),
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "seed": self.seed,
            "device": self.device,
            "game": asdict(self.game),
            "training": asdict(self.training),
            "model": asdict(self.model),
            "algorithm": asdict(self.algorithm),
        }

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"AionConfig(name='{self.name}', "
            f"game={self.game.name}, "
            f"algo={'VR-' if self.algorithm.use_vr else ''}{self.algorithm.scheduler_type.upper()}, "
            f"iters={self.training.iterations})"
        )


# Preset configurations for common experiments
def leduc_vr_ddcfr_config() -> AionConfig:
    """SOTA configuration: Leduc Poker with VR-DDCFR+."""
    return AionConfig(
        name="leduc_vr_ddcfr",
        game=GameConfig(name="leduc"),
        training=TrainingConfig(iterations=1000, batch_size=128, buffer_capacity=5000),
        model=ModelConfig(hidden_size=128, num_hidden_layers=4, learning_rate=0.001),
        algorithm=AlgorithmConfig(use_vr=True, scheduler_type="ddcfr", gamma=2.0),
    )


def kuhn_vanilla_config() -> AionConfig:
    """Baseline configuration: Kuhn Poker with standard Deep CFR."""
    return AionConfig(
        name="kuhn_vanilla",
        game=GameConfig(name="kuhn"),
        training=TrainingConfig(iterations=500, batch_size=64, buffer_capacity=1000),
        model=ModelConfig(hidden_size=64, num_hidden_layers=3, learning_rate=0.001),
        algorithm=AlgorithmConfig(use_vr=False, scheduler_type="linear", gamma=1.0),
    )


def river_holdem_config() -> AionConfig:
    """Texas Hold'em River endgame solving with VR-PDCFR+.

    Configured for 52-card poker with larger buffer and batch sizes.
    Uses head-to-head evaluation instead of NashConv.
    """
    return AionConfig(
        name="river_holdem",
        game=GameConfig(name="river_holdem"),
        training=TrainingConfig(
            iterations=10000,
            batch_size=1024,
            buffer_capacity=100000,
            eval_every=1000,
            log_every=100
        ),
        model=ModelConfig(hidden_size=128, num_hidden_layers=3, learning_rate=0.001),
        algorithm=AlgorithmConfig(
            use_vr=True,
            scheduler_type="pdcfr",
            alpha=2.0,
            beta=0.5,
            gamma=1.0
        ),
    )
