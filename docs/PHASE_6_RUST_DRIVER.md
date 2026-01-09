# Phase 6: Rust Driver Architecture - Disk-Native Design

**Goal**: Eliminate Python<->Rust boundary crossing overhead AND solve the divergence problem caused by reservoir sampling's "forgetting" behavior.

---

## The Problem: Reservoir Sampling Causes Divergence

### Experimental Evidence

**Linear Weighting Run**:
- Peak: 1.5M (+435 mbb/h)
- Collapse: 2.5M (-469 mbb/h)

**Uniform Weighting Run**:
- Peak: 2M (+690 mbb/h)
- Collapse: 4M (-243 mbb/h)

**Root Cause**: Reservoir sampling deletes old samples to make room for new ones. This causes:
1. **Strategy Inconsistency**: Network "forgets" early game situations
2. **Foundation Erosion**: We delete the "foundations" to build the "roof"
3. **Catastrophic Forgetting**: Average strategy becomes unstable

### The Insight

> "We are not limited by RAM. We are limited by Architecture."

Deep CFR requires **all historical samples** to compute a stable average strategy. Reservoir sampling fundamentally violates this requirement.

---

## The Solution: Disk-Native Streaming

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Rust Driver (Collection)                                         │
│                                                                  │
│  for epoch in 0..num_epochs:                                    │
│      writer = TrajectoryWriter::new(f"data/epoch_{epoch}.bin") │
│                                                                  │
│      for traversal in 0..traversals_per_epoch:                  │
│          state, target = traverse(game)                         │
│          writer.append(state, target)  // Disk write           │
│                                                                  │
│      writer.flush()                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Binary files on disk
┌─────────────────────────────────────────────────────────────────┐
│ data/                                                            │
│   epoch_0.bin   (50K samples × 140 floats × 4 bytes = 28 MB)   │
│   epoch_1.bin                                                    │
│   epoch_2.bin                                                    │
│   ...                                                            │
│   epoch_99.bin                                                   │
│                                                                  │
│   Total: 100 epochs × 28 MB = 2.8 GB (for 5M samples)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Memory-mapped access
┌─────────────────────────────────────────────────────────────────┐
│ Python Training (Consumption)                                    │
│                                                                  │
│  dataset = TrajectoryDataset("data/")  # mmap all .bin files   │
│                                                                  │
│  for batch in DataLoader(dataset, batch_size=4096):             │
│      loss = train_step(batch)                                   │
│                                                                  │
│  # Zero RAM: Only current batch in memory                       │
│  # Zero Forgetting: All samples accessible                      │
│  # Infinite Scale: Can grow to 100M samples                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Binary File Format

### Record Layout

Each sample is a fixed-size binary record:

```
┌────────────────────────────────────────────────────────────┐
│ State Encoding (136 × f32 = 544 bytes)                     │
├────────────────────────────────────────────────────────────┤
│ Target/Regret (4 × f32 = 16 bytes)                        │
└────────────────────────────────────────────────────────────┘
Total: 560 bytes per sample
```

### File Structure

```
epoch_N.bin:
  [Record 0: 560 bytes]
  [Record 1: 560 bytes]
  [Record 2: 560 bytes]
  ...
  [Record K: 560 bytes]

File size: K × 560 bytes
Number of samples: file_size / 560
```

### Why Binary?

| Format | Size/Sample | Read Speed | Random Access |
|--------|-------------|------------|---------------|
| JSON | ~2 KB | Slow (parse) | No |
| CSV | ~600 bytes | Medium | No |
| **Binary** | **560 bytes** | **Fast** | **Yes (mmap)** |
| HDF5 | ~580 bytes | Fast | Yes |

Binary is simplest and fastest. No dependencies, no parsing overhead.

---

## Rust Implementation

### TrajectoryWriter

```rust
// src/aion26_rust/src/io.rs

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write, Result};
use std::path::Path;

/// Binary trajectory writer for disk-native Deep CFR
///
/// Writes (state, target) pairs to disk in a compact binary format.
/// Each record is 560 bytes: 136 × f32 (state) + 4 × f32 (target)
pub struct TrajectoryWriter {
    writer: BufWriter<File>,
    samples_written: usize,
}

impl TrajectoryWriter {
    /// Create a new writer for the given path
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;

        Ok(Self {
            writer: BufWriter::with_capacity(1024 * 1024, file),  // 1MB buffer
            samples_written: 0,
        })
    }

    /// Append a (state, target) pair to the file
    ///
    /// Args:
    ///     state: 136-dimensional state encoding
    ///     target: 4-dimensional regret target
    pub fn append(&mut self, state: &[f32; 136], target: &[f32; 4]) -> Result<()> {
        // Write state (136 × 4 = 544 bytes)
        let state_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                state.as_ptr() as *const u8,
                136 * std::mem::size_of::<f32>()
            )
        };
        self.writer.write_all(state_bytes)?;

        // Write target (4 × 4 = 16 bytes)
        let target_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                target.as_ptr() as *const u8,
                4 * std::mem::size_of::<f32>()
            )
        };
        self.writer.write_all(target_bytes)?;

        self.samples_written += 1;
        Ok(())
    }

    /// Flush the buffer and sync to disk
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()
    }

    /// Get number of samples written
    pub fn len(&self) -> usize {
        self.samples_written
    }
}

impl Drop for TrajectoryWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
```

### RustTrainer with Cooperative Inference + Disk Streaming

The key insight is that we need **cooperative inference**: Rust traverses the game tree but
pauses when it needs neural network predictions. Python runs GPU inference and returns
predictions. Rust resumes traversal.

```rust
// src/aion26_rust/src/trainer.rs - Cooperative Inference Protocol

/// Result of calling step() - tells Python what to do next
#[pyclass]
pub enum StepResult {
    /// Need inference: Rust has `count` queries waiting in buffer
    RequestInference { count: usize },

    /// Batch finished: `samples_added` samples written to disk
    Finished { samples_added: usize },
}

/// Traversal stage in the state machine
enum TraversalStage {
    PreInference,           // Need network prediction
    PostInference { ... },  // Exploring actions
    Finalizing { ... },     // Computing regrets
    WaitingForChild,        // Chance node waiting for child
}

#[pyclass]
pub struct RustTrainer {
    // Disk storage
    data_dir: PathBuf,
    writer: Option<TrajectoryWriter>,

    // Query buffer for batched inference
    query_buffer: QueryBuffer,

    // Stack machine for pause/resume traversal
    context_stack: Vec<TraversalContext>,

    // Prediction cache (query_id -> strategy)
    prediction_cache: HashMap<usize, Vec<f32>>,
}

#[pymethods]
impl RustTrainer {
    /// Execute one step of cooperative traversal
    ///
    /// Args:
    ///     inference_results: Predictions from previous RequestInference
    ///     num_traversals: Number of traversals (required on first call)
    ///
    /// Returns:
    ///     StepResult::RequestInference if Rust needs predictions
    ///     StepResult::Finished when all traversals complete
    pub fn step(
        &mut self,
        inference_results: Option<&PyArray2<f32>>,
        num_traversals: Option<usize>,
    ) -> PyResult<StepResult> {
        // Distribute inference results to waiting contexts
        if let Some(predictions) = inference_results {
            self.distribute_predictions(predictions)?;
        }

        // Process contexts until we need inference or finish
        loop {
            if !self.context_stack.is_empty() {
                match self.resume_top_context()? {
                    ResumeResult::NeedInference => {
                        return Ok(StepResult::RequestInference {
                            count: self.query_buffer.count,
                        });
                    }
                    ResumeResult::MadeProgress => continue,
                    ResumeResult::ContextPopped => continue,
                }
            }

            // Start new traversal if needed
            if self.completed_traversals < self.target_traversals {
                self.start_new_traversal()?;
                continue;
            }

            break;
        }

        Ok(StepResult::Finished { samples_added: self.epoch_samples })
    }

    /// Get query buffer for network inference (zero-copy view)
    pub fn get_query_buffer(&self, py: Python) -> PyResult<PyArray2<f32>> { ... }
}
```

### Python Orchestration Loop

```python
# scripts/train_phase6.py - The cooperative inference loop

def run_epoch_generation(trainer, network, device, num_traversals):
    """Run the generation phase for one epoch.

    This is the cooperative inference loop:
    1. Rust traverses game tree
    2. When it needs predictions, returns RequestInference
    3. Python runs batch inference on GPU
    4. Loop back with predictions until Finished
    """
    network.eval()
    predictions = None

    while True:
        result = trainer.step(predictions, num_traversals=num_traversals if predictions is None else None)

        if result.is_finished():
            break

        elif result.is_request_inference():
            # Get query buffer from Rust (zero-copy view)
            query_buffer = trainer.get_query_buffer()

            # Run batch inference on GPU
            with torch.no_grad():
                queries_tensor = torch.from_numpy(query_buffer).to(device)
                predictions_tensor = network(queries_tensor)

            # Convert back to NumPy for Rust
            predictions = predictions_tensor.cpu().numpy()

    return result.count()  # samples written to disk
```

---

## Python Dataset Implementation

### TrajectoryDataset (Memory-Mapped)

```python
# src/aion26/data/trajectory_dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import mmap

class TrajectoryDataset(Dataset):
    """Memory-mapped dataset for disk-native Deep CFR.

    Reads binary trajectory files without loading them into RAM.
    Supports random access for efficient batch sampling.
    """

    RECORD_SIZE = 560  # 136 + 4 floats × 4 bytes
    STATE_SIZE = 136
    TARGET_SIZE = 4

    def __init__(self, data_dir: str, epochs: list[int] = None):
        """
        Args:
            data_dir: Directory containing epoch_*.bin files
            epochs: List of epoch indices to include (None = all)
        """
        self.data_dir = Path(data_dir)

        # Find all epoch files
        if epochs is None:
            self.files = sorted(self.data_dir.glob("epoch_*.bin"))
        else:
            self.files = [self.data_dir / f"epoch_{i}.bin" for i in epochs]
            self.files = [f for f in self.files if f.exists()]

        # Compute cumulative sample counts
        self.file_sizes = []
        self.cumulative_sizes = [0]

        for f in self.files:
            size = f.stat().st_size
            num_samples = size // self.RECORD_SIZE
            self.file_sizes.append(num_samples)
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + num_samples)

        self.total_samples = self.cumulative_sizes[-1]

        # Memory-map all files
        self.mmaps = []
        for f in self.files:
            with open(f, 'rb') as fp:
                mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
                self.mmaps.append(mm)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Find which file contains this index
        file_idx = 0
        for i, cumsize in enumerate(self.cumulative_sizes[1:], 1):
            if idx < cumsize:
                file_idx = i - 1
                break

        # Compute offset within file
        local_idx = idx - self.cumulative_sizes[file_idx]
        byte_offset = local_idx * self.RECORD_SIZE

        # Read record from mmap
        mm = self.mmaps[file_idx]
        record = mm[byte_offset:byte_offset + self.RECORD_SIZE]

        # Parse state and target
        state = np.frombuffer(record[:self.STATE_SIZE * 4], dtype=np.float32)
        target = np.frombuffer(record[self.STATE_SIZE * 4:], dtype=np.float32)

        return torch.from_numpy(state.copy()), torch.from_numpy(target.copy())

    def __del__(self):
        for mm in self.mmaps:
            mm.close()


class WeightedEpochSampler:
    """Samples batches with recency weighting across epochs.

    More recent epochs get higher weight, but old epochs are never forgotten.
    """

    def __init__(self, dataset: TrajectoryDataset, batch_size: int, alpha: float = 1.0):
        """
        Args:
            dataset: TrajectoryDataset
            batch_size: Samples per batch
            alpha: Recency weight (0 = uniform, 1 = linear, 2 = quadratic)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.alpha = alpha

        # Compute epoch weights
        num_epochs = len(dataset.files)
        weights = [(i + 1) ** alpha for i in range(num_epochs)]
        total = sum(weights)
        self.epoch_probs = [w / total for w in weights]

    def sample_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch with recency weighting."""
        states = []
        targets = []

        for _ in range(self.batch_size):
            # Sample epoch according to weights
            epoch_idx = np.random.choice(len(self.epoch_probs), p=self.epoch_probs)

            # Sample uniformly within epoch
            start = self.dataset.cumulative_sizes[epoch_idx]
            end = self.dataset.cumulative_sizes[epoch_idx + 1]
            sample_idx = np.random.randint(start, end)

            state, target = self.dataset[sample_idx]
            states.append(state)
            targets.append(target)

        return torch.stack(states), torch.stack(targets)
```

---

## Training Loop

### Disk-Native Training Script

```python
#!/usr/bin/env python3
"""Train Deep CFR with disk-native sample storage."""

import torch
from aion26_rust import RustTrainer
from aion26.data.trajectory_dataset import TrajectoryDataset, WeightedEpochSampler
from aion26.deep_cfr.networks import DeepCFRNetwork

# Config
NUM_EPOCHS = 100
TRAVERSALS_PER_EPOCH = 50_000  # 100K samples per epoch (2 players)
TRAIN_STEPS_PER_EPOCH = 1000
BATCH_SIZE = 4096

# Initialize
trainer = RustTrainer(data_dir="data/trajectories")
network = DeepCFRNetwork(input_size=136, output_size=4, hidden_size=512)
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

for epoch in range(NUM_EPOCHS):
    # === 1. COLLECTION PHASE (Rust) ===
    print(f"Epoch {epoch}: Collecting {TRAVERSALS_PER_EPOCH} traversals...")
    trainer.start_epoch()
    trainer.collect_experience(TRAVERSALS_PER_EPOCH)
    samples = trainer.end_epoch()
    print(f"  Wrote {samples} samples to epoch_{epoch}.bin")

    # === 2. TRAINING PHASE (PyTorch) ===
    print(f"Epoch {epoch}: Training on ALL historical data...")

    # Load dataset (memory-mapped, zero RAM)
    dataset = TrajectoryDataset("data/trajectories")
    sampler = WeightedEpochSampler(dataset, BATCH_SIZE, alpha=1.0)

    total_loss = 0
    for step in range(TRAIN_STEPS_PER_EPOCH):
        states, targets = sampler.sample_batch()

        predictions = network(states)
        loss = torch.nn.functional.mse_loss(predictions, targets / 500.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / TRAIN_STEPS_PER_EPOCH
    print(f"  Loss: {avg_loss:.4f} | Total samples: {trainer.total_samples():,}")

    # Cleanup mmap
    del dataset, sampler

print(f"\nTraining complete. {trainer.total_samples():,} samples on disk.")
```

---

## Benefits of Disk-Native Architecture

### 1. Zero Forgetting
- Every sample ever generated is preserved
- Historical consistency maintained
- No catastrophic forgetting

### 2. Zero RAM Pressure
- Only current batch (4096 × 560 = 2.2 MB) in memory
- Can scale to billions of samples
- Works on any machine regardless of RAM

### 3. Reproducibility
- Samples are deterministically stored
- Can replay training from any checkpoint
- Can analyze sample distribution offline

### 4. Weighted Replay
- Can weight recent epochs higher (like Linear)
- But never delete old epochs (unlike Reservoir)
- Best of both worlds: recency bias + full history

### 5. Fault Tolerance
- Samples on disk survive crashes
- Can resume training from any epoch
- Can parallelize collection across machines

---

## Storage Requirements

| Training Size | Samples | Disk Space |
|--------------|---------|------------|
| 1M traversals | 2M | 1.1 GB |
| 5M traversals | 10M | 5.5 GB |
| 50M traversals | 100M | 55 GB |
| 500M traversals | 1B | 550 GB |

Modern SSDs: 1-2 GB/s sequential write. 5M samples = 2.8 GB = 2-3 seconds.

---

## Implementation Checklist

### Phase 6.1: Disk I/O ✅
- [x] Implement `TrajectoryWriter` in Rust (`src/aion26_rust/src/io.rs`)
- [x] Binary format: 560 bytes/sample (136 state + 4 target × f32)
- [x] Buffered writes (1MB buffer)

### Phase 6.2: RustTrainer with Cooperative Inference ✅
- [x] Implement `StepResult` enum (RequestInference, Finished)
- [x] Implement `TraversalStage` state machine (PreInference, PostInference, Finalizing, WaitingForChild)
- [x] Implement `TraversalContext` stack for pause/resume semantics
- [x] Implement `QueryBuffer` for batched inference
- [x] Implement `step()` cooperative protocol
- [x] Fix External Sampling (opponent samples ONE action)
- [x] Fix chance node handling (WaitingForChild stage)

### Phase 6.3: Python Dataset ✅
- [x] Implement `TrajectoryDataset` with mmap (`src/aion26/memory/disk.py`)
- [x] Implement `WeightedEpochSampler` with recency bias
- [x] Implement `UniformEpochSampler` for uniform sampling
- [x] Unit tests: 12 tests passing (`tests/test_disk_native.py`)

### Phase 6.4: Training Integration ✅
- [x] Create `train_phase6.py` orchestrator script
- [x] Implement `run_epoch_generation()` cooperative loop
- [x] Implement `run_epoch_training()` with disk streaming
- [x] Smoke test: 2 epochs, 357 samples, 61 samples/s

### Phase 6.5: Validation (In Progress)
- [ ] Run 5M training with disk storage
- [ ] Verify monotonic improvement (no divergence)
- [ ] Compare to reservoir sampling baselines

**Status**: Core implementation complete. Ready for validation run.

---

## Open Questions

### 1. SSD vs HDD Performance?
SSD strongly preferred (100× faster random access). HDD works but training will be slower.

### 2. Compression?
Could use LZ4 compression (2-3× space savings). Adds CPU overhead but reduces I/O. Defer to Phase 7.

### 3. Parallel Collection?
Could run multiple Rust workers writing to different epoch files. Defer to Phase 7.

### 4. Cloud Storage?
Could stream to S3/GCS for infinite scale. Adds latency. Defer to Phase 7.
