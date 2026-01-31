# Webapp Enhancement Plan

## **Current State Assessment**

### **Architecture Overview**
**Aion-26** is a well-structured Deep PDCFR+ (Predictive Discounted CFR) framework for solving imperfect information games:

**Core strengths:**
- ✅ **Solid fundamentals**: 3 phases complete (Vanilla CFR → Deep CFR → Deep PDCFR+)
- ✅ **Modern ML stack**: PyTorch, Flask/SocketIO, numpy
- ✅ **Test coverage**: 13 test files, 87% coverage
- ✅ **Performance**: 45K-55K samples/s, 95% improvement over vanilla
- ✅ **Modular design**: Clean separation (games/, cfr/, deep_cfr/, learner/, metrics/)
- ✅ **Webapp exists**: Two training dashboards already implemented (`train_webapp.py`, `train_webapp_enhanced.py`)

### **Code Quality (Best Practices)**

**Excellent:**
- Clean Python architecture with protocols (duck typing)
- Type hints throughout (`GameState(Protocol)`)
- Comprehensive docstrings (NumPy style)
- Configuration management (YAML configs)
- Factory patterns for schedulers
- Proper separation of concerns (learner/memory/metrics)

**Good:**
- Git workflow (meaningful commits, clean history)
- Structured documentation (phase reports, specs)
- Baseline comparisons for validation

**Areas for improvement:**
- No experiment tracking (WandB mentioned but not integrated)
- Single-seed evaluation (robustness not validated)
- Minimal error handling in webapp
- No logging framework (uses print statements)
- No deployment infrastructure

### **Webapp Status**

**You already have TWO webapps!**

1. **`train_webapp.py`** (basic):
   - Real-time training dashboard
   - Throughput metrics (samples/sec)
   - Loss curves
   - Simple, functional

2. **`train_webapp_enhanced.py`** (advanced):
   - Everything from basic +
   - Win rate tracking (mbb/h)
   - Action distribution visualization
   - Recent hands log
   - Performance summary
   - Polyak-averaged training
   - Historical mixing

**Current gaps:**
- No **strategy inspection** (can't view learned policies)
- No **hyperparameter tuning UI**
- No **comparison dashboard** (vanilla vs PDCFR+)
- No **model management** (save/load/compare checkpoints)
- No **dataset management** (epoch files scattered)
- No **evaluation suite** (vs baselines)
- Limited **error handling/recovery**

---

## **Strategic Plan: Production-Ready Webapp**

### **Phase 4: Web Dashboard & Model Management**

**Goals:**
1. **Unified dashboard** for all training/eval/comparison workflows
2. **Model registry** for checkpoint management
3. **Evaluation suite** with baseline comparisons
4. **Strategy inspector** to visualize learned policies
5. **Production-grade infrastructure** (logging, error handling, Docker)

### **Implementation Roadmap**

#### **1. Enhanced Dashboard (Week 1-2)**

**Merge existing webapps into unified experience:**

```
/                    → Landing page (overview, quick start)
/train               → Training dashboard (enhanced version)
/eval                → Evaluation suite (baseline comparisons)
/inspect             → Strategy inspector (policy visualization)
/compare             → Side-by-side model comparison
/models              → Model registry (checkpoints, metadata)
```

**New features:**
- **Multi-run tracking**: Track multiple training runs simultaneously
- **Hyperparameter panel**: Tune α/β/learning rate on the fly
- **Strategy heatmaps**: Visualize policy evolution for key info sets
- **Exploitability tracking**: Real-time NashConv computation
- **Baseline evaluation**: Auto-evaluate against RandomBot/CallingStation
- **Model diffing**: Compare two checkpoints visually

#### **2. Backend Improvements (Week 2-3)**

**Infrastructure:**
```python
# Structured logging
import structlog
logger = structlog.get_logger()

# Database for experiment tracking (SQLite → PostgreSQL)
from sqlalchemy import create_engine
# Store: runs, checkpoints, metrics, evaluations

# Celery for async tasks
from celery import Celery
app = Celery('aion26', broker='redis://localhost:6379')

@app.task
def train_model(config):
    # Long-running training task
    pass

@app.task
def evaluate_model(model_id, baseline):
    # Evaluation task
    pass
```

**API endpoints:**
```python
POST /api/runs/create               # Start new training run
GET  /api/runs/{id}                 # Get run status
POST /api/runs/{id}/stop            # Stop training
GET  /api/runs/{id}/metrics         # Get metrics history
POST /api/models/{id}/evaluate      # Evaluate against baselines
GET  /api/models                    # List all checkpoints
POST /api/models/compare            # Compare two models
GET  /api/strategy/{model_id}/{infoset}  # Get strategy for info set
```

#### **3. Strategy Inspector (Week 3)**

**Key feature: Visualize learned policies**

```python
# New endpoint: GET /api/strategy/{model_id}/{info_set}
def get_strategy(model_id: str, info_set: str) -> dict:
    model = load_model(model_id)
    state_encoding = encode_info_set(info_set)

    with torch.no_grad():
        advantages = model(state_encoding)

    strategy = regret_matching(advantages)

    return {
        'info_set': info_set,
        'actions': ['fold', 'call', 'raise', 'allin'],
        'probabilities': strategy.tolist(),
        'advantages': advantages.tolist(),
    }
```

**UI visualization:**
- **Interactive tree**: Click through game tree, see policy at each node
- **Heatmap**: Strategy distribution across info sets
- **Action breakdown**: Fold/call/raise frequencies by street
- **Exploitability**: Identify weak spots in policy

#### **4. Evaluation Suite (Week 4)**

**Automated benchmarking:**
```python
class EvaluationSuite:
    def __init__(self, model):
        self.model = model
        self.baselines = {
            'random': RandomBot(),
            'calling_station': CallingStationBot(),
            'always_fold': AlwaysFoldBot(),
            'always_raise': AlwaysRaiseBot(),
        }

    def run_all(self, num_hands=10000):
        results = {}
        for name, baseline in self.baselines.items():
            mbb = self.evaluate_vs(baseline, num_hands)
            results[name] = {
                'mbb': mbb,
                'hands': num_hands,
                'win_rate': mbb / 1000 * 2,  # Convert mbb to bb/hand
            }
        return results
```

**Dashboard display:**
- **Win rate table**: mbb/h vs each baseline
- **Confidence intervals**: Bootstrap 95% CI
- **Hand replays**: View interesting hands
- **Strategy evolution**: How policy changed over training

#### **5. Production Infrastructure (Week 5)**

**Docker deployment:**
```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5001
CMD ["gunicorn", "-k", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker", "-w", "1", "-b", "0.0.0.0:5001", "app:app"]
```

**docker-compose.yml:**
```yaml
services:
  webapp:
    build: .
    ports:
      - "5001:5001"
    volumes:
      - ./data:/app/data
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@db:5432/aion26

  redis:
    image: redis:7-alpine

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: aion26

  celery:
    build: .
    command: celery -A app.celery worker --loglevel=info
    depends_on:
      - redis
      - db
```

**Monitoring:**
- **Prometheus metrics**: Request latency, training throughput
- **Grafana dashboards**: System health
- **Sentry**: Error tracking

---

## **Recommended Next Steps**

### **Option A: Quick Win (1-2 days)** ✅ SELECTED
Enhance existing `train_webapp_enhanced.py` with:
1. Strategy inspector (show top-K info sets with probabilities)
2. Model save/load UI (dropdown to load checkpoints)
3. Baseline comparison (add RandomBot/CallingStation eval)
4. Better error handling + logging

**Deliverable**: Production-ready single-page dashboard

### **Option B: Full Platform (2-3 weeks)**
Build comprehensive webapp with:
1. Multi-page dashboard (train/eval/inspect/compare)
2. Database-backed experiment tracking
3. Async task queue (Celery)
4. Model registry with metadata
5. Docker deployment
6. API documentation (Swagger)

**Deliverable**: Enterprise-grade ML platform

### **Option C: Research Focus (1 week)**
Keep webapp simple, focus on algorithmic improvements:
1. Multi-seed validation (test robustness)
2. Hyperparameter sweep (α/β grid search)
3. WandB integration (experiment tracking)
4. Regret Matching+ implementation
5. Importance sampling

**Deliverable**: Publication-quality results

---

## **Implementation Plan: Option A**

**Week 1**: Enhance existing webapp
- Add strategy inspector
- Add baseline evaluation
- Add model save/load
- Add proper logging

**Week 2**: Infrastructure
- Add SQLite for experiment tracking
- Add Redis + Celery for async tasks
- Add API layer

**Week 3**: Advanced features
- Multi-run tracking
- Model comparison
- Hyperparameter tuning UI

**Week 4**: Production polish
- Docker deployment
- Error handling
- Monitoring
- Documentation

This approach delivers immediate value (working dashboard in days) while setting up for long-term scalability.
