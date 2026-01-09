"""Reservoir buffer for experience replay in Deep CFR.

OPTIMIZED TENSOR IMPLEMENTATION:
- Pre-allocated tensors (not Python lists)
- O(1) sampling with torch.randint (not O(N) random.sample)
- Circular buffer with reservoir sampling

Performance:
- Old: O(N) sampling, degraded to 120 it/s at 2M buffer
- New: O(1) sampling, constant 500-1000 it/s regardless of buffer size
"""

import torch
from typing import Tuple, Optional


class ReservoirBuffer:
    """Reservoir buffer with O(1) tensor operations.

    Uses pre-allocated tensors and torch.randint for constant-time sampling.

    Attributes:
        capacity: Maximum number of samples to store
        input_size: Dimension of state tensors
        output_size: Dimension of target tensors
        device: Torch device (cpu or cuda)
        states: Tensor storage [capacity, input_size]
        targets: Tensor storage [capacity, output_size]
        ptr: Current write pointer (circular)
        size: Current number of samples in buffer
        total_seen: Total samples added (for reservoir sampling)
    """

    def __init__(
        self,
        capacity: int,
        input_shape: tuple,
        output_size: int = 4,
        device: str = "cpu"
    ):
        """Initialize tensor-based reservoir buffer.

        Args:
            capacity: Maximum number of samples to store
            input_shape: Shape of state tensors (e.g., (136,) for River)
            output_size: Dimension of targets (default: 4 actions)
            device: Torch device ("cpu" or "cuda")

        Raises:
            ValueError: If capacity <= 0
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self.capacity = capacity
        self.input_shape = input_shape
        self.output_size = output_size
        self.device = torch.device(device)

        # Extract input size from shape
        if isinstance(input_shape, int):
            self.input_size = input_shape
        elif len(input_shape) == 1:
            self.input_size = input_shape[0]
        else:
            raise ValueError(f"Only 1D input shapes supported, got {input_shape}")

        # Pre-allocate tensors (O(1) indexing, no Python list overhead)
        self.states = torch.zeros(
            (capacity, self.input_size),
            dtype=torch.float32,
            device=self.device
        )
        self.targets = torch.zeros(
            (capacity, output_size),
            dtype=torch.float32,
            device=self.device
        )

        # Buffer state
        self.ptr = 0  # Write pointer (circular)
        self.size = 0  # Current number of samples
        self.total_seen = 0  # Total samples added (for reservoir sampling)

    def add(self, state: torch.Tensor, target: torch.Tensor):
        """Add a sample to the buffer.

        Uses reservoir sampling when buffer is full:
        - For first k samples: store directly
        - For sample n > k: replace random slot with probability k/n

        Args:
            state: State tensor of shape (input_size,)
            target: Target tensor of shape (output_size,)

        Complexity: O(1)
        """
        # Move to device if needed
        if state.device != self.device:
            state = state.to(self.device)
        if target.device != self.device:
            target = target.to(self.device)

        self.total_seen += 1

        # Phase 1: Fill buffer sequentially
        if self.size < self.capacity:
            idx = self.size
            self.states[idx] = state
            self.targets[idx] = target
            self.size += 1
        else:
            # Phase 2: Reservoir sampling
            # Generate random index in [0, total_seen)
            # If < capacity, replace that slot
            # This gives each sample equal probability k/n of being in buffer
            j = torch.randint(0, self.total_seen, (1,)).item()
            if j < self.capacity:
                self.states[j] = state
                self.targets[j] = target

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch from the buffer.

        CRITICAL SPEEDUP: Uses torch.randint (O(1)) instead of
        random.sample(range(N), k) which is O(N).

        Args:
            batch_size: Number of samples to return

        Returns:
            Tuple of (states, targets):
            - states: Tensor of shape (batch_size, input_size)
            - targets: Tensor of shape (batch_size, output_size)

        Raises:
            ValueError: If batch_size > buffer size or buffer is empty

        Complexity: O(1) for index generation + O(batch_size) for gathering
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        if batch_size > self.size:
            raise ValueError(
                f"Batch size {batch_size} exceeds buffer size {self.size}"
            )

        # O(1) random index generation (THE FIX)
        indices = torch.randint(
            0, self.size, (batch_size,),
            device=self.device
        )

        # O(batch_size) tensor indexing (fast, no Python loop)
        states_batch = self.states[indices]
        targets_batch = self.targets[indices]

        return states_batch, targets_batch

    def __len__(self) -> int:
        """Return current number of samples in buffer."""
        return self.size

    def __repr__(self) -> str:
        """String representation of buffer."""
        return (
            f"ReservoirBuffer(capacity={self.capacity}, "
            f"size={self.size}, "
            f"total_seen={self.total_seen}, "
            f"device={self.device})"
        )

    def clear(self):
        """Clear all samples from buffer and reset counters."""
        self.size = 0
        self.ptr = 0
        self.total_seen = 0
        # Don't need to zero tensors (will be overwritten)

    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all samples in buffer.

        Returns:
            Tuple of (states, targets) containing all buffered samples

        Raises:
            ValueError: If buffer is empty
        """
        if self.size == 0:
            raise ValueError("Buffer is empty")

        # Return only filled portion
        return self.states[:self.size], self.targets[:self.size]

    @property
    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        return self.size >= self.capacity

    @property
    def fill_percentage(self) -> float:
        """Get percentage of buffer currently filled."""
        return (self.size / self.capacity) * 100.0
