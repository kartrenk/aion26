"""Disk-native trajectory storage for Deep CFR.

This module provides memory-mapped access to binary trajectory files,
enabling training on datasets larger than RAM without forgetting.

Key insight: Deep CFR NEEDS all historical samples. Reservoir sampling
causes catastrophic forgetting. Solution: Store everything on disk.

File Format:
- Each record: 560 bytes (136 × f32 state + 4 × f32 target)
- Files named: epoch_{N}.bin
- Number of samples = file_size / 560
"""

from __future__ import annotations

import mmap
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


# Must match Rust constants in io.rs
STATE_DIM = 136
TARGET_DIM = 4
RECORD_SIZE = (STATE_DIM + TARGET_DIM) * 4  # 560 bytes


class EpochFile:
    """Memory-mapped view of a single epoch file."""

    def __init__(self, path: Path):
        self.path = path
        self.file_size = path.stat().st_size
        self.num_samples = self.file_size // RECORD_SIZE

        # Memory-map the file for zero-copy reads
        self._file = open(path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        # Create a numpy view (zero-copy)
        self._data = np.frombuffer(self._mmap, dtype=np.float32)
        self._data = self._data.reshape(self.num_samples, STATE_DIM + TARGET_DIM)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get a single (state, target) sample."""
        record = self._data[idx]
        state = record[:STATE_DIM]
        target = record[STATE_DIM:]
        return state, target

    def get_batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get a batch of samples efficiently."""
        records = self._data[indices]
        states = records[:, :STATE_DIM]
        targets = records[:, STATE_DIM:]
        return states, targets

    def close(self):
        """Release resources."""
        # Clear numpy view first to release buffer
        del self._data
        try:
            self._mmap.close()
        except BufferError:
            # Buffer still held by numpy view - will be released on GC
            pass
        self._file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TrajectoryDataset(Dataset):
    """PyTorch Dataset over all epoch files in a directory.

    Provides unified access to samples across multiple epoch files
    with memory-mapped loading (zero RAM usage).

    Usage:
        dataset = TrajectoryDataset("data/trajectories")
        loader = DataLoader(
            dataset,
            batch_sampler=WeightedEpochSampler(dataset, batch_size=4096),
        )
        for states, targets in loader:
            loss = train_step(states, targets)
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.epoch_files: list[EpochFile] = []
        self.epoch_offsets: list[int] = []  # Cumulative sample counts

        self._load_epoch_files()

    def _load_epoch_files(self):
        """Scan directory for epoch files and load them."""
        # Find all epoch files, sorted by epoch number
        epoch_paths = sorted(
            self.data_dir.glob("epoch_*.bin"), key=lambda p: int(p.stem.split("_")[1])
        )

        cumulative = 0
        for path in epoch_paths:
            ef = EpochFile(path)
            if ef.num_samples > 0:
                self.epoch_files.append(ef)
                self.epoch_offsets.append(cumulative)
                cumulative += ef.num_samples

        self._total_samples = cumulative

    def refresh(self):
        """Reload epoch files (call after new epochs are written)."""
        # Close existing files
        for ef in self.epoch_files:
            ef.close()
        self.epoch_files.clear()
        self.epoch_offsets.clear()
        self._load_epoch_files()

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample by global index."""
        # Binary search to find which epoch file
        epoch_idx = self._find_epoch(idx)
        ef = self.epoch_files[epoch_idx]
        local_idx = idx - self.epoch_offsets[epoch_idx]

        state, target = ef[local_idx]
        return torch.from_numpy(state.copy()), torch.from_numpy(target.copy())

    def _find_epoch(self, global_idx: int) -> int:
        """Find which epoch file contains the given global index."""
        # Binary search
        lo, hi = 0, len(self.epoch_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.epoch_offsets[mid] <= global_idx:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def get_batch_from_indices(self, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of samples from global indices.

        More efficient than individual __getitem__ calls when indices
        are contiguous within epoch files.
        """
        # Group indices by epoch file for efficient batch access
        states_list = []
        targets_list = []

        # Sort indices and track original positions for reordering
        sorted_order = np.argsort(indices)
        sorted_indices = indices[sorted_order]

        current_epoch = -1
        epoch_indices = []

        for i, idx in enumerate(sorted_indices):
            epoch_idx = self._find_epoch(idx)
            if epoch_idx != current_epoch:
                # Process previous epoch's indices
                if epoch_indices:
                    self._fetch_epoch_batch(current_epoch, epoch_indices, states_list, targets_list)
                current_epoch = epoch_idx
                epoch_indices = []

            local_idx = idx - self.epoch_offsets[epoch_idx]
            epoch_indices.append(local_idx)

        # Process final epoch
        if epoch_indices:
            self._fetch_epoch_batch(current_epoch, epoch_indices, states_list, targets_list)

        # Concatenate and reorder to original index order
        states = np.concatenate(states_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)

        # Reorder to match original indices
        inverse_order = np.argsort(sorted_order)
        states = states[inverse_order]
        targets = targets[inverse_order]

        return torch.from_numpy(states), torch.from_numpy(targets)

    def _fetch_epoch_batch(
        self,
        epoch_idx: int,
        local_indices: list[int],
        states_list: list[np.ndarray],
        targets_list: list[np.ndarray],
    ):
        """Fetch a batch from a single epoch file."""
        ef = self.epoch_files[epoch_idx]
        indices_arr = np.array(local_indices, dtype=np.int64)
        states, targets = ef.get_batch(indices_arr)
        states_list.append(states.copy())
        targets_list.append(targets.copy())

    @property
    def num_epochs(self) -> int:
        return len(self.epoch_files)

    @property
    def samples_per_epoch(self) -> list[int]:
        return [ef.num_samples for ef in self.epoch_files]

    def close(self):
        """Release all resources."""
        for ef in self.epoch_files:
            ef.close()
        self.epoch_files.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class WeightedEpochSampler(Sampler[list[int]]):
    """Batch sampler with recency weighting across epochs.

    Samples with recency bias: newer epochs get higher probability.
    This provides the benefits of linear weighting without forgetting.

    Weight formula: weight(epoch) = alpha^(num_epochs - epoch - 1)
    - alpha=1.0: Uniform sampling (all epochs equal)
    - alpha=0.5: Newest epoch gets 2x weight of previous
    - alpha=0.25: Newest epoch gets 4x weight of previous

    Usage:
        sampler = WeightedEpochSampler(dataset, batch_size=4096, alpha=0.5)
        loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        dataset: TrajectoryDataset,
        batch_size: int,
        alpha: float = 0.8,  # Reduced from 0.5 to prevent epoch overfitting
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.alpha = alpha  # 0.8 = gentle recency (1.25x per epoch), 0.5 = aggressive (2x)
        self.drop_last = drop_last

        self._compute_weights()

    def _compute_weights(self):
        """Compute sampling weights for each epoch."""
        num_epochs = self.dataset.num_epochs
        if num_epochs == 0:
            self._weights = np.array([])
            self._cumulative_weights = np.array([])
            return

        # Recency weights: newer epochs get higher weight
        epoch_weights = np.array([self.alpha ** (num_epochs - i - 1) for i in range(num_epochs)])

        # Scale by number of samples in each epoch
        samples_per_epoch = np.array(self.dataset.samples_per_epoch)
        self._weights = epoch_weights * samples_per_epoch
        self._weights /= self._weights.sum()  # Normalize

        # For sampling: cumulative weights
        self._cumulative_weights = np.cumsum(self._weights)

    def refresh(self):
        """Refresh weights after dataset changes."""
        self._compute_weights()

    def __iter__(self) -> Iterator[list[int]]:
        """Generate batches of global indices."""
        total_samples = len(self.dataset)
        if total_samples == 0:
            return

        num_batches = total_samples // self.batch_size
        if not self.drop_last and total_samples % self.batch_size != 0:
            num_batches += 1

        for _ in range(num_batches):
            # Sample epoch indices according to weights
            epoch_samples = np.random.random(self.batch_size)
            epoch_indices = np.searchsorted(self._cumulative_weights, epoch_samples)
            epoch_indices = np.clip(epoch_indices, 0, self.dataset.num_epochs - 1)

            # Sample within each epoch
            batch_indices = []
            for epoch_idx in epoch_indices:
                ef = self.dataset.epoch_files[epoch_idx]
                local_idx = np.random.randint(0, ef.num_samples)
                global_idx = self.dataset.epoch_offsets[epoch_idx] + local_idx
                batch_indices.append(global_idx)

            yield batch_indices

    def __len__(self) -> int:
        total_samples = len(self.dataset)
        if self.drop_last:
            return total_samples // self.batch_size
        return (total_samples + self.batch_size - 1) // self.batch_size


class UniformEpochSampler(WeightedEpochSampler):
    """Uniform sampling across all epochs (no recency bias).

    Equivalent to WeightedEpochSampler with alpha=1.0.
    """

    def __init__(
        self,
        dataset: TrajectoryDataset,
        batch_size: int,
        drop_last: bool = False,
    ):
        super().__init__(dataset, batch_size, alpha=1.0, drop_last=drop_last)
