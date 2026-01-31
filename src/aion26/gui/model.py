"""Training backend for GUI with queue-based communication.

Runs Deep CFR training in a separate thread and sends metrics back to the main thread
via a queue to avoid freezing the GUI.
"""

from __future__ import annotations

import threading
import queue
import logging
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

from aion26.config import AionConfig
from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.deep_cfr.networks import KuhnEncoder, LeducEncoder, HoldemEncoder
from aion26.games.kuhn import new_kuhn_game
from aion26.games.leduc import LeducPoker
from aion26.games.river_holdem import new_river_holdem_game
from aion26.metrics.exploitability import compute_nash_conv
from aion26.learner.discounting import (
    PDCFRScheduler,
    LinearScheduler,
    DDCFRStrategyScheduler,
)


@dataclass
class MetricsUpdate:
    """Metrics update sent from training thread to GUI."""

    iteration: int
    loss: float
    value_loss: float
    buffer_size: int
    buffer_fill_pct: float
    nash_conv: Optional[float] = None  # Only computed periodically
    strategy: Optional[dict[str, npt.NDArray[np.float64]]] = None
    status: Literal["training", "completed", "error"] = "training"
    error_message: Optional[str] = None


class TrainingThread(threading.Thread):
    """Background thread for Deep CFR training.

    Accepts an AionConfig, initializes the trainer, and runs the training loop
    while sending metrics back through a queue.

    Example:
        >>> config = AionConfig(...)
        >>> metrics_queue = queue.Queue()
        >>> thread = TrainingThread(config, metrics_queue)
        >>> thread.start()
        >>> # In main thread
        >>> while True:
        ...     metrics = metrics_queue.get()
        ...     print(f"Iteration {metrics.iteration}: Loss={metrics.loss}")
        ...     if metrics.status == "completed":
        ...         break
    """

    def __init__(
        self,
        config: AionConfig,
        metrics_queue: queue.Queue[MetricsUpdate],
        stop_event: Optional[threading.Event] = None,
    ):
        """Initialize training thread.

        Args:
            config: Experiment configuration
            metrics_queue: Queue to send metrics updates to GUI
            stop_event: Optional event to signal training should stop
        """
        super().__init__(daemon=True)
        self.config = config
        self.metrics_queue = metrics_queue
        self.stop_event = stop_event or threading.Event()
        self.trainer: Optional[DeepCFRTrainer] = None
        logger.info(
            f"TrainingThread created: game={config.game.name}, algo={config.algorithm.scheduler_type}, iters={config.training.iterations}"
        )

    def _initialize_trainer(self):
        """Initialize the Deep CFR trainer based on config."""
        logger.info("Initializing trainer...")
        # Create initial state based on game
        if self.config.game.name == "kuhn":
            initial_state = new_kuhn_game()
            encoder = KuhnEncoder()
            input_size = encoder.input_size
            output_size = 2  # Check or Bet
            logger.info(f"Game: Kuhn Poker, input_size={input_size}, output_size={output_size}")
        elif self.config.game.name == "leduc":
            initial_state = LeducPoker()
            encoder = LeducEncoder()
            input_size = encoder.input_size
            output_size = 3  # Fold, Check, or Bet
            logger.info(f"Game: Leduc Poker, input_size={input_size}, output_size={output_size}")
        elif self.config.game.name == "river_holdem":
            initial_state = new_river_holdem_game()
            encoder = HoldemEncoder()
            input_size = encoder.input_size
            output_size = 4  # Fold, Check/Call, Bet Pot, All-In
            logger.info(f"Game: River Hold'em, input_size={input_size}, output_size={output_size}")
        else:
            raise ValueError(f"Unknown game: {self.config.game.name}")

        # Create discounting schedulers
        if self.config.algorithm.scheduler_type == "uniform":
            # Vanilla CFR - no discounting at all
            regret_scheduler = None
            strategy_scheduler = None
            logger.info("Using UNIFORM scheduler (vanilla CFR - no discounting)")
        elif self.config.algorithm.scheduler_type == "linear":
            regret_scheduler = None  # No discounting
            strategy_scheduler = LinearScheduler()
            logger.info("Using LINEAR scheduler")
        elif self.config.algorithm.scheduler_type == "pdcfr":
            regret_scheduler = PDCFRScheduler(
                alpha=self.config.algorithm.alpha,
                beta=self.config.algorithm.beta,
            )
            strategy_scheduler = LinearScheduler()
            logger.info(
                f"Using PDCFR scheduler (α={self.config.algorithm.alpha}, β={self.config.algorithm.beta})"
            )
        elif self.config.algorithm.scheduler_type == "ddcfr":
            regret_scheduler = PDCFRScheduler(
                alpha=self.config.algorithm.alpha,
                beta=self.config.algorithm.beta,
            )
            strategy_scheduler = DDCFRStrategyScheduler(gamma=self.config.algorithm.gamma)
            logger.info(
                f"Using DDCFR scheduler (α={self.config.algorithm.alpha}, β={self.config.algorithm.beta}, γ={self.config.algorithm.gamma})"
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.algorithm.scheduler_type}")

        # Initialize trainer
        self.trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=input_size,
            output_size=output_size,
            hidden_size=self.config.model.hidden_size,
            num_hidden_layers=self.config.model.num_hidden_layers,
            buffer_capacity=self.config.training.buffer_capacity,
            learning_rate=self.config.model.learning_rate,
            batch_size=self.config.training.batch_size,
            seed=self.config.seed,
            device=self.config.device,
            regret_scheduler=regret_scheduler,
            strategy_scheduler=strategy_scheduler,
        )

    def _get_average_strategy(self) -> dict[str, npt.NDArray[np.float64]]:
        """Get the current average strategy from the trainer."""
        if self.trainer is None:
            return {}

        strategy = {}
        for info_state in self.trainer.strategy_sum.keys():
            strategy[info_state] = self.trainer.get_average_strategy(info_state)

        return strategy

    def run(self):
        """Run the training loop (executed in background thread)."""
        try:
            # Initialize trainer
            logger.info("Starting training thread...")
            self._initialize_trainer()
            if self.trainer is None:
                raise RuntimeError("Failed to initialize trainer")
            logger.info("Trainer initialized successfully")

            # Training loop
            logger.info(f"Starting training loop for {self.config.training.iterations} iterations")
            for i in range(self.config.training.iterations):
                # Check if we should stop
                if self.stop_event.is_set():
                    logger.info(f"Training stopped at iteration {i}")
                    self.metrics_queue.put(
                        MetricsUpdate(
                            iteration=i,
                            loss=0.0,
                            value_loss=0.0,
                            buffer_size=0,
                            buffer_fill_pct=0.0,
                            status="completed",
                        )
                    )
                    return

                # Run one iteration
                metrics = self.trainer.run_iteration()

                # Log progress every 10 iterations
                if (i + 1) % 10 == 0:
                    logger.debug(
                        f"Iter {i + 1}: loss={metrics['loss']:.4f}, buffer={metrics['buffer_size']}/{self.config.training.buffer_capacity}"
                    )

                # Compute NashConv periodically (skip for river_holdem - infeasible)
                nash_conv = None
                strategy = None
                if (i + 1) % self.config.training.eval_every == 0:
                    strategy = self._get_average_strategy()

                    # Only compute NashConv for small games (Kuhn, Leduc)
                    # For River Hold'em, use head-to-head evaluation instead
                    if self.config.game.name in ["kuhn", "leduc"]:
                        logger.info(f"Computing NashConv at iteration {i + 1}...")
                        nash_conv = compute_nash_conv(self.trainer.initial_state, strategy)
                        logger.info(f"NashConv at iteration {i + 1}: {nash_conv:.6f}")
                    else:
                        logger.info(
                            f"Skipping NashConv for {self.config.game.name} (use head-to-head evaluation instead)"
                        )
                        nash_conv = None

                # Send metrics to GUI every iteration for real-time progress
                # (but only include strategy/nashconv when computed)
                update = MetricsUpdate(
                    iteration=metrics["iteration"],
                    loss=metrics["loss"],
                    value_loss=metrics["value_loss"],
                    buffer_size=metrics["buffer_size"],
                    buffer_fill_pct=metrics["buffer_fill_pct"],
                    nash_conv=nash_conv,
                    strategy=strategy,
                    status="training",
                )
                self.metrics_queue.put(update)

            # Final update
            logger.info("Training completed")
            final_strategy = self._get_average_strategy()

            # Only compute final NashConv for small games (Kuhn, Leduc)
            # For River Hold'em, use head-to-head evaluation instead
            if self.config.game.name in ["kuhn", "leduc"]:
                logger.info("Computing final NashConv...")
                final_nash_conv = compute_nash_conv(self.trainer.initial_state, final_strategy)
                logger.info(f"Final NashConv: {final_nash_conv:.6f}")
            else:
                logger.info(
                    f"Skipping final NashConv for {self.config.game.name} (use head-to-head evaluation instead)"
                )
                final_nash_conv = None
            self.metrics_queue.put(
                MetricsUpdate(
                    iteration=self.config.training.iterations,
                    loss=self.trainer.get_average_loss(),
                    value_loss=0.0,
                    buffer_size=len(self.trainer.buffer),
                    buffer_fill_pct=self.trainer.buffer.fill_percentage,
                    nash_conv=final_nash_conv,
                    strategy=final_strategy,
                    status="completed",
                )
            )
            logger.info("Training thread completed successfully")

        except Exception as e:
            # Send error to GUI
            logger.exception(f"Training thread crashed: {e}")
            self.metrics_queue.put(
                MetricsUpdate(
                    iteration=0,
                    loss=0.0,
                    value_loss=0.0,
                    buffer_size=0,
                    buffer_fill_pct=0.0,
                    status="error",
                    error_message=str(e),
                )
            )
            raise

    def stop(self):
        """Signal the training thread to stop."""
        self.stop_event.set()
