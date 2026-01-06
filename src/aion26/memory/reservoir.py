"""Reservoir buffer for experience replay in Deep CFR.

Implements reservoir sampling to maintain a uniform sample of training data
across all iterations, ensuring older samples aren't systematically discarded.

Algorithm:
    For the first k samples: Store directly
    For sample n (n > k):
        - Generate random j in [0, n)
        - If j < k, replace buffer[j] with new sample
        - Otherwise discard new sample

This ensures each sample has equal probability k/n of being in the buffer.
"""

import torch
import random
from typing import Tuple, Optional


class ReservoirBuffer:
    """Reservoir buffer for storing training samples with uniform sampling.

    Uses reservoir sampling to maintain a fixed-size buffer where every
    sample ever seen has equal probability of being in the buffer.

    Attributes:
        capacity: Maximum number of samples to store
        input_shape: Shape of state tensors (excluding batch dimension)
        total_seen: Total number of samples added (including discarded ones)
        states: List of stored state tensors
        targets: List of stored target tensors
    """

    def __init__(self, capacity: int, input_shape: tuple):
        """Initialize reservoir buffer.

        Args:
            capacity: Maximum number of samples to store
            input_shape: Shape of state tensors (e.g., (10,) for Kuhn)

        Raises:
            ValueError: If capacity <= 0
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self.capacity = capacity
        self.input_shape = input_shape
        self.total_seen = 0

        # Storage for states and targets
        self.states = []
        self.targets = []

    def add(self, state: torch.Tensor, target: torch.Tensor):
        """Add a sample to the buffer using reservoir sampling.

        Args:
            state: State tensor of shape input_shape
            target: Target tensor (e.g., regrets for actions)

        Raises:
            ValueError: If state shape doesn't match input_shape
        """
        # Validate state shape
        if state.shape != self.input_shape:
            raise ValueError(
                f"State shape {state.shape} doesn't match "
                f"expected shape {self.input_shape}"
            )

        self.total_seen += 1

        # Phase 1: Fill buffer to capacity
        if len(self.states) < self.capacity:
            self.states.append(state.clone().detach())
            self.targets.append(target.clone().detach())
        else:
            # Phase 2: Reservoir sampling
            # Generate random index j in [0, total_seen)
            j = random.randint(0, self.total_seen - 1)

            # If j < capacity, replace buffer[j] with new sample
            if j < self.capacity:
                self.states[j] = state.clone().detach()
                self.targets[j] = target.clone().detach()
            # Otherwise, discard new sample (implicitly)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch from the buffer.

        Args:
            batch_size: Number of samples to return

        Returns:
            Tuple of (states, targets) as stacked tensors:
            - states: Tensor of shape (batch_size, *input_shape)
            - targets: Tensor of shape (batch_size, *target_shape)

        Raises:
            ValueError: If batch_size > buffer size
            ValueError: If buffer is empty
        """
        if len(self.states) == 0:
            raise ValueError("Cannot sample from empty buffer")

        if batch_size > len(self.states):
            raise ValueError(
                f"Batch size {batch_size} exceeds buffer size {len(self.states)}"
            )

        # Sample random indices without replacement
        indices = random.sample(range(len(self.states)), batch_size)

        # Stack selected samples
        states_batch = torch.stack([self.states[i] for i in indices])
        targets_batch = torch.stack([self.targets[i] for i in indices])

        return states_batch, targets_batch

    def __len__(self) -> int:
        """Return current number of samples in buffer."""
        return len(self.states)

    def __repr__(self) -> str:
        """String representation of buffer."""
        return (
            f"ReservoirBuffer(capacity={self.capacity}, "
            f"size={len(self)}, "
            f"total_seen={self.total_seen})"
        )

    def clear(self):
        """Clear all samples from buffer and reset counters."""
        self.states.clear()
        self.targets.clear()
        self.total_seen = 0

    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all samples in buffer as stacked tensors.

        Returns:
            Tuple of (states, targets) containing all buffered samples

        Raises:
            ValueError: If buffer is empty
        """
        if len(self.states) == 0:
            raise ValueError("Buffer is empty")

        states_all = torch.stack(self.states)
        targets_all = torch.stack(self.targets)

        return states_all, targets_all

    @property
    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        return len(self.states) >= self.capacity

    @property
    def fill_percentage(self) -> float:
        """Get percentage of buffer currently filled."""
        return (len(self.states) / self.capacity) * 100.0
