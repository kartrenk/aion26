# Phase 6: Inference Bridge - Batch Query Protocol

**Problem**: Rust Driver needs network predictions to guide CFR traversal, but calling Python per node destroys performance.

**Solution**: Cooperative execution with batch inference protocol. Rust pauses when it needs predictions, Python runs batch inference, Rust resumes with results.

---

## The Bottleneck: Per-Node Python Calls

### Naive Approach (BROKEN)
```
Rust: traverse() {
    for node in game_tree:
        let strategy = python_network_predict(state)  ← 1-5ms per call
        let action = sample(strategy)
        ...
}
```

**Cost**: 5-10 nodes per game × 100 games × 1ms = **500-1000ms per training step** (10× slower than target)

---

## The Solution: Query Buffer + Batch Inference

### Cooperative Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│ Python Training Loop                                             │
│                                                                  │
│  for step in range(num_steps):                                  │
│      while True:                                                 │
│          result = trainer.step(inference_results)               │
│                                                                  │
│          match result:                                           │
│              StepResult.RequestInference(count):                │
│                  # Rust hit the query buffer limit              │
│                  queries = trainer.get_query_buffer()           │
│                  predictions = network.predict(queries)  # GPU  │
│                  # Loop back with predictions                   │
│                                                                  │
│              StepResult.Finished(samples):                      │
│                  # Batch complete, train network                │
│                  break                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Details

### 1. Query Buffer (Rust Side)

**Purpose**: Accumulate network queries during traversal without blocking.

```rust
struct QueryBuffer {
    states: Vec<f32>,        // Flattened: max_queries × 136 floats
    node_ids: Vec<usize>,    // Which traversal node is waiting
    max_queries: usize,      // Batch size (e.g., 4096)
    count: usize,            // Current number of queries
}

impl QueryBuffer {
    fn add_query(&mut self, state: &[f32], node_id: usize) -> bool {
        if self.count >= self.max_queries {
            return false;  // Buffer full, need inference
        }

        let offset = self.count * 136;
        self.states[offset..offset + 136].copy_from_slice(state);
        self.node_ids.push(node_id);
        self.count += 1;
        true
    }

    fn clear(&mut self) {
        self.count = 0;
        self.node_ids.clear();
    }
}
```

### 2. Continuation Context (Rust Side)

**Purpose**: Save traversal state when pausing for inference.

```rust
struct TraversalContext {
    state: RustRiverHoldem,
    update_player: u8,
    reach_prob_0: f32,
    reach_prob_1: f32,
    phase: TraversalPhase,
}

enum TraversalPhase {
    ComputingCounterfactuals {
        action_idx: usize,
        partial_values: Vec<f32>,
    },
    SamplingOpponent {
        query_id: usize,  // Waiting for this query result
    },
}
```

**Idea**: When Rust needs inference, it saves the current traversal state to a stack, adds a query to the buffer, and moves to the next traversal. When Python returns predictions, Rust pops the stack and resumes.

### 3. Step Result (Python Interface)

```rust
#[pyclass]
#[derive(Clone)]
pub enum StepResult {
    /// Inference needed: Rust has accumulated `count` queries in buffer
    RequestInference { count: usize },

    /// Batch finished: `samples_added` samples added to replay buffer
    Finished { samples_added: usize },
}

#[pymethods]
impl StepResult {
    fn is_request_inference(&self) -> bool {
        matches!(self, StepResult::RequestInference { .. })
    }

    fn is_finished(&self) -> bool {
        matches!(self, StepResult::Finished { .. })
    }

    fn count(&self) -> usize {
        match self {
            StepResult::RequestInference { count } => *count,
            StepResult::Finished { samples_added } => *samples_added,
        }
    }
}
```

---

## RustTrainer API

### Core Methods

```rust
#[pyclass]
pub struct RustTrainer {
    // Existing fields
    game: RustRiverHoldem,
    traverser: CFRTraverser,
    buffer: ReservoirBuffer,

    // NEW: Inference protocol fields
    query_buffer: QueryBuffer,
    pending_contexts: Vec<TraversalContext>,
    current_traversals: usize,
    target_traversals: usize,
}

#[pymethods]
impl RustTrainer {
    /// Execute one step of cooperative traversal
    ///
    /// Args:
    ///     inference_results: Optional predictions from previous RequestInference
    ///                        Shape: (N, 4) where N matches previous request count
    ///
    /// Returns:
    ///     StepResult enum:
    ///         - RequestInference: Rust needs network predictions
    ///         - Finished: Batch complete
    pub fn step(
        &mut self,
        inference_results: Option<&PyArray2<f32>>,
    ) -> PyResult<StepResult> {
        // 1. If inference_results provided, distribute to pending contexts
        if let Some(predictions) = inference_results {
            self.distribute_predictions(predictions)?;
        }

        // 2. Resume pending traversals
        while let Some(ctx) = self.pending_contexts.pop() {
            let query_needed = self.resume_traversal(ctx)?;
            if query_needed {
                // Hit query buffer limit
                return Ok(StepResult::RequestInference {
                    count: self.query_buffer.count,
                });
            }
        }

        // 3. Start new traversals until target reached or buffer full
        while self.current_traversals < self.target_traversals {
            let query_needed = self.start_new_traversal()?;
            if query_needed {
                return Ok(StepResult::RequestInference {
                    count: self.query_buffer.count,
                });
            }
            self.current_traversals += 1;
        }

        // 4. Batch complete
        Ok(StepResult::Finished {
            samples_added: self.current_traversals * 2,  // 2 players
        })
    }

    /// Get query buffer for network inference (zero-copy view)
    pub fn get_query_buffer<'py>(
        &self,
        py: Python<'py>,
    ) -> &'py PyArray2<f32> {
        let count = self.query_buffer.count;
        let slice = &self.query_buffer.states[0..count * 136];

        // Zero-copy view into Rust memory
        PyArray2::from_slice(py, slice, (count, 136))
    }
}
```

---

## Python Training Loop

### Example Integration

```python
import aion26_rust
import torch
import numpy as np

# Setup
trainer = aion26_rust.RustTrainer(buffer_capacity=4_000_000)
network = load_advantage_network()  # PyTorch model

batch_states = np.zeros((4096, 136), dtype=np.float32)
batch_targets = np.zeros((4096, 4), dtype=np.float32)

# Training loop
for step in range(100_000):
    # === 1. COOPERATIVE TRAVERSAL (Rust + Python Network) ===
    inference_results = None

    while True:
        result = trainer.step(inference_results)

        if result.is_request_inference():
            # Rust needs network predictions
            count = result.count()

            # Zero-copy: view Rust's query buffer
            queries = trainer.get_query_buffer()  # Shape: (count, 136)

            # Batch inference on GPU
            with torch.no_grad():
                queries_tensor = torch.from_numpy(queries)
                predictions = network(queries_tensor)  # Shape: (count, 4)

            # Convert to NumPy for Rust
            inference_results = predictions.cpu().numpy()

            # Loop back to Rust with results
            continue

        elif result.is_finished():
            # Batch complete
            samples_added = result.count()
            break

    # === 2. TRAIN NETWORK (GPU) ===
    trainer.fill_batch(batch_states, batch_targets, batch_size=4096)
    states_tensor = torch.from_numpy(batch_states)
    targets_tensor = torch.from_numpy(batch_targets)

    loss = train_step(network, states_tensor, targets_tensor)

    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss:.4f} | Samples: {samples_added}")
```

---

## Performance Analysis

### Query Buffer Sizing

**Trade-off**: Larger buffer = fewer Python calls, but more memory + latency.

| Buffer Size | Calls/Step | Latency | Memory |
|-------------|------------|---------|--------|
| 512 | ~20 | Low | 256 KB |
| 2048 | ~5 | Medium | 1 MB |
| 4096 | ~2-3 | Medium | 2 MB |
| 8192 | ~1-2 | High | 4 MB |

**Recommendation**: 4096 (matches training batch size, ~2-3 inference calls per step)

### Expected Throughput

**Baseline (Python Driver)**: 900 trav/s
- 7,500 Python<->Rust crossings per step
- ~1.1 ms per step

**Phase 6 (Rust Driver + Batch Inference)**: 3,000-5,000 trav/s
- 2-3 Python<->Rust crossings per step
- Network inference: 2-3 × 5ms = 10-15ms
- Rust traversal: ~5ms
- Total: ~20ms per step → 50 steps/s → 2,500 trav/s (@ 50 trav/step)

**Speedup**: 3-5× vs Python driver

---

## Alternative Approaches (Rejected)

### 1. Move Network to Rust (libtorch)

**Pros**: Zero Python calls during traversal
**Cons**:
- Complex (C++ bindings, tensor management)
- Loses PyTorch ecosystem (debugging, logging, WandB)
- Hard to update network weights from Python

**Decision**: Rejected for Phase 6, consider for Phase 7

### 2. Async Python Calls (asyncio)

**Pros**: Non-blocking inference
**Cons**:
- Python GIL still blocks (can't parallelize with Rust)
- Complex state management
- Minimal benefit vs cooperative protocol

**Decision**: Rejected

### 3. Separate Inference Process (IPC)

**Pros**: True parallelism (Rust + Python run concurrently)
**Cons**:
- IPC overhead (sockets/pipes)
- Complex synchronization
- Debugging nightmare

**Decision**: Rejected for Phase 6

---

## Implementation Plan

### Phase 6.1: Query Buffer (2 hours)
- [ ] Implement QueryBuffer struct
- [ ] Add to RustTrainer
- [ ] Test buffer overflow detection

### Phase 6.2: Continuation Context (4 hours)
- [ ] Implement TraversalContext stack
- [ ] Modify traverse() to pause/resume
- [ ] Test context save/restore correctness

### Phase 6.3: Step Method (3 hours)
- [ ] Implement StepResult enum
- [ ] Implement step() method
- [ ] Implement get_query_buffer()

### Phase 6.4: Integration Test (2 hours)
- [ ] Python training loop with mock network
- [ ] Verify zero-copy query buffer access
- [ ] Verify convergence matches Python driver

### Phase 6.5: Performance Benchmark (1 hour)
- [ ] Measure trav/s with real PyTorch network
- [ ] Compare vs Python driver baseline
- [ ] Profile bottlenecks

**Total**: 12 hours

---

## Success Criteria

### Functional
- [ ] Convergence matches Python driver (±5% exploitability)
- [ ] No memory leaks (100K steps with Valgrind)
- [ ] Correct regret computation (unit tests vs tabular CFR)

### Performance
- [ ] Throughput ≥2,500 trav/s (3× Python driver)
- [ ] <3 inference calls per training step
- [ ] Query buffer zero-copy verified (pointer equality check)

### Integration
- [ ] Works with existing PyTorch networks
- [ ] Easy to debug (clear error messages on protocol violations)
- [ ] Backward compatible (can fall back to collect_experience())

---

## Open Questions

### 1. Regret Matching in Rust or Python?

**Option A (Rust)**: Apply regret matching in Rust after getting advantage predictions
```rust
fn regret_matching(advantages: &[f32]) -> Vec<f32> {
    let positive_regrets: Vec<f32> = advantages.iter()
        .map(|&a| a.max(0.0))
        .collect();
    let sum: f32 = positive_regrets.iter().sum();
    if sum > 0.0 {
        positive_regrets.iter().map(|&r| r / sum).collect()
    } else {
        vec![0.25; 4]  // Uniform fallback
    }
}
```
**Pros**: Faster, keeps logic in Rust
**Cons**: Duplicates Python logic

**Option B (Python)**: Network outputs final strategy (softmax in network)
**Pros**: Single source of truth
**Cons**: Slower (extra forward pass)

**Decision**: **Option A** - Rust regret matching (Performance > DRY)

### 2. Parallel Query Buffer Filling?

Could use Rayon to fill query buffer in parallel across multiple traversals.

**Benefit**: Saturate buffer faster, fewer inference calls
**Risk**: Complex synchronization, harder to debug

**Decision**: Sequential for Phase 6, parallel for Phase 7

### 3. Dynamic Buffer Sizing?

Auto-adjust buffer size based on network latency?

**Benefit**: Optimal trade-off without manual tuning
**Risk**: Non-deterministic performance, complex heuristics

**Decision**: Fixed 4096 for Phase 6

---

## Appendix: Step Method Implementation

### Detailed Pseudocode

```rust
pub fn step(&mut self, inference_results: Option<&PyArray2<f32>>) -> PyResult<StepResult> {
    // STEP 1: Distribute inference results to pending contexts
    if let Some(predictions) = inference_results {
        assert_eq!(predictions.shape()[0], self.query_buffer.count);

        // Map query_id -> prediction
        let mut pred_map = HashMap::new();
        for (i, node_id) in self.query_buffer.node_ids.iter().enumerate() {
            let pred_slice = predictions.row(i);  // 4 floats
            pred_map.insert(*node_id, pred_slice.to_vec());
        }

        // Apply predictions to pending contexts
        for ctx in &mut self.pending_contexts {
            if let TraversalPhase::SamplingOpponent { query_id } = ctx.phase {
                if let Some(strategy) = pred_map.get(&query_id) {
                    // Resume this context with strategy
                    ctx.phase = TraversalPhase::Completed { strategy };
                }
            }
        }

        // Clear query buffer
        self.query_buffer.clear();
    }

    // STEP 2: Resume pending traversals
    while let Some(mut ctx) = self.pending_contexts.pop() {
        match self.continue_traversal(&mut ctx) {
            ContinueResult::Finished => {
                // Traversal complete, sample added to buffer
            }
            ContinueResult::QueryNeeded => {
                // Hit buffer limit, push context back
                self.pending_contexts.push(ctx);
                return Ok(StepResult::RequestInference {
                    count: self.query_buffer.count,
                });
            }
        }
    }

    // STEP 3: Start new traversals
    while self.current_traversals < self.target_traversals {
        let mut ctx = TraversalContext::new(
            self.game.clone(),
            self.current_traversals % 2,  // Alternate players
        );

        match self.continue_traversal(&mut ctx) {
            ContinueResult::Finished => {
                self.current_traversals += 1;
            }
            ContinueResult::QueryNeeded => {
                self.pending_contexts.push(ctx);
                return Ok(StepResult::RequestInference {
                    count: self.query_buffer.count,
                });
            }
        }
    }

    // STEP 4: Batch complete
    let samples_added = self.current_traversals * 2;
    self.current_traversals = 0;  // Reset for next batch

    Ok(StepResult::Finished { samples_added })
}
```

### Safety Invariants

1. **Query Buffer Consistency**: `query_buffer.count == query_buffer.node_ids.len()`
2. **Context Stack Bounded**: `pending_contexts.len() <= target_traversals`
3. **No Dangling Queries**: All `query_id` in contexts exist in buffer
4. **Prediction Count Match**: `inference_results.shape()[0] == query_buffer.count`

These invariants are checked with `debug_assert!` in debug builds.
