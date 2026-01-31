"""Integration test for disk-native Deep CFR pipeline.

Tests the complete flow:
1. RustTrainer writes trajectories to binary files
2. TrajectoryDataset reads via memory-mapping
3. WeightedEpochSampler provides recency-biased batches
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import aion26_rust
from aion26.memory.disk import (
    TrajectoryDataset,
    WeightedEpochSampler,
    UniformEpochSampler,
    STATE_DIM,
    TARGET_DIM,
    RECORD_SIZE,
)


def collect_samples_with_step(trainer, num_traversals: int) -> int:
    """Helper to collect samples using the cooperative step() API.

    Uses uniform predictions (0.25 each) for testing purposes.
    Returns number of samples collected.
    """
    predictions = None

    while True:
        result = trainer.step(
            predictions, num_traversals=num_traversals if predictions is None else None
        )

        if result.is_finished():
            return result.count()

        elif result.is_request_inference():
            # Get number of queries
            query_count = trainer.query_buffer_count()

            # Provide uniform predictions (0.25 each action)
            predictions = np.full((query_count, TARGET_DIM), 0.0, dtype=np.float32)

            # Reset num_traversals since we already started
            num_traversals = None


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    d = tempfile.mkdtemp(prefix="dcfr_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestRustTrainerDiskNative:
    """Test RustTrainer disk-native storage."""

    def test_trainer_creation(self, temp_data_dir):
        """Test that trainer creates the data directory."""
        trainer = aion26_rust.RustTrainer(temp_data_dir)
        assert Path(temp_data_dir).exists()
        assert trainer.current_epoch() == 0
        assert trainer.total_samples() == 0

    def test_epoch_lifecycle(self, temp_data_dir):
        """Test start_epoch/step/end_epoch cycle."""
        trainer = aion26_rust.RustTrainer(temp_data_dir)

        # Start epoch 0
        trainer.start_epoch(0)
        assert trainer.current_epoch() == 0

        # Collect some experience (50 traversals)
        samples = collect_samples_with_step(trainer, 50)
        # Samples vary based on game tree exploration, just check > 0
        assert samples > 0

        # End epoch
        epoch_samples = trainer.end_epoch()
        assert epoch_samples > 0

        # Verify file was created
        epoch_file = Path(temp_data_dir) / "epoch_0.bin"
        assert epoch_file.exists()
        assert epoch_file.stat().st_size == samples * RECORD_SIZE

    def test_multiple_epochs(self, temp_data_dir):
        """Test writing multiple epochs."""
        trainer = aion26_rust.RustTrainer(temp_data_dir)

        for epoch in range(3):
            trainer.start_epoch(epoch)
            collect_samples_with_step(trainer, 10)  # 10 traversals per epoch
            trainer.end_epoch()

        assert trainer.total_samples() > 0  # Samples written across epochs

        # Verify all files exist
        for epoch in range(3):
            epoch_file = Path(temp_data_dir) / f"epoch_{epoch}.bin"
            assert epoch_file.exists()


class TestTrajectoryDataset:
    """Test TrajectoryDataset memory-mapped reading."""

    def test_empty_directory(self, temp_data_dir):
        """Test with no epoch files."""
        dataset = TrajectoryDataset(temp_data_dir)
        assert len(dataset) == 0
        assert dataset.num_epochs == 0

    def test_single_epoch(self, temp_data_dir):
        """Test reading a single epoch file."""
        # Create data with RustTrainer
        trainer = aion26_rust.RustTrainer(temp_data_dir)
        trainer.start_epoch(0)
        samples = collect_samples_with_step(trainer, 50)  # ~50 traversals
        trainer.end_epoch()

        # Load with dataset
        dataset = TrajectoryDataset(temp_data_dir)
        assert len(dataset) == samples
        assert dataset.num_epochs == 1

        # Access individual samples
        state, target = dataset[0]
        assert state.shape == (STATE_DIM,)
        assert target.shape == (TARGET_DIM,)
        assert state.dtype == torch.float32

    def test_multiple_epochs(self, temp_data_dir):
        """Test reading multiple epoch files."""
        trainer = aion26_rust.RustTrainer(temp_data_dir)

        for epoch in range(3):
            trainer.start_epoch(epoch)
            collect_samples_with_step(trainer, 10)  # 10 traversals per epoch
            trainer.end_epoch()

        dataset = TrajectoryDataset(temp_data_dir)
        assert len(dataset) > 0
        assert dataset.num_epochs == 3
        assert len(dataset.samples_per_epoch) == 3

        # Access sample from each epoch (use relative indices based on actual samples)
        for epoch_idx in range(3):
            if dataset.samples_per_epoch[epoch_idx] > 0:
                state, target = dataset[dataset.epoch_offsets[epoch_idx]]
                assert state.shape == (STATE_DIM,)
                assert target.shape == (TARGET_DIM,)

    def test_batch_access(self, temp_data_dir):
        """Test efficient batch access."""
        trainer = aion26_rust.RustTrainer(temp_data_dir)
        trainer.start_epoch(0)
        samples = collect_samples_with_step(trainer, 100)  # 100 traversals
        trainer.end_epoch()

        dataset = TrajectoryDataset(temp_data_dir)
        # Create indices that are valid for the actual sample count
        max_idx = len(dataset) - 1
        indices = np.array([0, min(10, max_idx), min(50, max_idx)])
        states, targets = dataset.get_batch_from_indices(indices)

        assert states.shape == (len(indices), STATE_DIM)
        assert targets.shape == (len(indices), TARGET_DIM)

    def test_refresh(self, temp_data_dir):
        """Test refreshing after new epoch is written."""
        trainer = aion26_rust.RustTrainer(temp_data_dir)

        # Write first epoch
        trainer.start_epoch(0)
        samples_0 = collect_samples_with_step(trainer, 10)
        trainer.end_epoch()

        # Load dataset
        dataset = TrajectoryDataset(temp_data_dir)
        assert len(dataset) == samples_0

        # Write second epoch
        trainer.start_epoch(1)
        samples_1 = collect_samples_with_step(trainer, 10)
        trainer.end_epoch()

        # Refresh and verify
        dataset.refresh()
        assert len(dataset) == samples_0 + samples_1
        assert dataset.num_epochs == 2


class TestWeightedEpochSampler:
    """Test weighted sampling across epochs."""

    def test_uniform_sampling(self, temp_data_dir):
        """Test uniform sampling with alpha=1.0."""
        trainer = aion26_rust.RustTrainer(temp_data_dir)
        for epoch in range(3):
            trainer.start_epoch(epoch)
            collect_samples_with_step(trainer, 50)  # ~50 traversals per epoch
            trainer.end_epoch()

        dataset = TrajectoryDataset(temp_data_dir)
        sampler = UniformEpochSampler(dataset, batch_size=32)

        # Generate batches
        batches = list(sampler)
        expected_batches = (len(dataset) + 31) // 32  # Ceiling division
        assert len(batches) == expected_batches

        # Each batch should have 32 indices (except possibly last)
        for batch in batches[:-1]:  # Last batch may be smaller
            assert len(batch) == 32

    def test_recency_weighted_sampling(self, temp_data_dir):
        """Test recency-weighted sampling."""
        trainer = aion26_rust.RustTrainer(temp_data_dir)
        for epoch in range(3):
            trainer.start_epoch(epoch)
            collect_samples_with_step(trainer, 50)  # ~50 traversals per epoch
            trainer.end_epoch()

        dataset = TrajectoryDataset(temp_data_dir)
        sampler = WeightedEpochSampler(dataset, batch_size=32, alpha=0.25)

        # Generate many batches and count epoch distribution
        epoch_counts = [0, 0, 0]
        for batch in sampler:
            for idx in batch:
                epoch_idx = dataset._find_epoch(idx)
                epoch_counts[epoch_idx] += 1

        # With alpha=0.25, newest epoch should have ~4x weight
        # Epoch 2 should have significantly more samples than epoch 0
        total = sum(epoch_counts)
        ratios = [c / total for c in epoch_counts]
        print(f"Epoch distribution: {epoch_counts} -> {ratios}")

        # Epoch 2 (newest) should have higher ratio than epoch 0 (oldest)
        assert ratios[2] > ratios[0]


class TestDataLoaderIntegration:
    """Test integration with PyTorch DataLoader."""

    def test_dataloader_iteration(self, temp_data_dir):
        """Test using DataLoader with WeightedEpochSampler."""
        trainer = aion26_rust.RustTrainer(temp_data_dir)
        for epoch in range(2):
            trainer.start_epoch(epoch)
            collect_samples_with_step(trainer, 25)  # ~25 traversals per epoch
            trainer.end_epoch()

        dataset = TrajectoryDataset(temp_data_dir)
        sampler = WeightedEpochSampler(dataset, batch_size=16, alpha=0.5)

        # Direct iteration over sampler (batch indices)
        batch_count = 0
        for batch_indices in sampler:
            indices = np.array(batch_indices)
            states, targets = dataset.get_batch_from_indices(indices)
            assert states.shape[1] == STATE_DIM
            assert targets.shape[1] == TARGET_DIM
            batch_count += 1

        assert batch_count > 0

    def test_with_standard_dataloader(self, temp_data_dir):
        """Test with standard DataLoader (single sample access)."""
        trainer = aion26_rust.RustTrainer(temp_data_dir)
        trainer.start_epoch(0)
        collect_samples_with_step(trainer, 25)  # ~25 traversals
        trainer.end_epoch()

        dataset = TrajectoryDataset(temp_data_dir)

        # Standard DataLoader with batch_size
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        batch_count = 0
        for states, targets in loader:
            assert states.shape[1] == STATE_DIM
            assert targets.shape[1] == TARGET_DIM
            batch_count += 1

        assert batch_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
