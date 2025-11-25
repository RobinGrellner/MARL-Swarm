# MARL-Swarm

## Key-Commands:
### Training:
#### TRPO
uv run python training/train_rendezvous.py --algorithm trpo --num-agents 20 --max-agents 100 --world-size 100 --comm-radius 75.0 --obs-model local_basic --total-timesteps 5000000 --num-vec-envs 8 --model-path models/huttenrauch
_trpo_20_agents_scalable.zip --tensorboard-log logs/trpo/huttenrauch_scalable --use-cuda --n-steps 256 --batch-size 128

#### PPO

### Evaluation:



**Multi-Agent Reinforcement Learning for Swarm Robotics Research**

This repository is part of my Master's thesis, implementing PettingZoo-compatible environments for training coordinated swarm behaviors using Multi-Agent Reinforcement Learning (MARL) with PPO and parameter sharing.

## Overview

MARL-Swarm provides a framework for training point-agent swarms with single- and double-integrator kinematics in 2D continuous spaces. The primary implemented scenario is **rendezvous**, where agents learn to minimize pairwise distances and converge to a common location.

### Key Features

- **Multiple environments**: Rendezvous (primary) and Pursuit-Evasion (implemented)
- **PettingZoo-compatible multi-agent RL** with parameter sharing
- **Multiple observation models** (6 variants from classic to advanced)
- **Scale-invariant policies** using mean embedding feature extractors
- **Flexible kinematics** (single/double integrator, configurable speeds)
- **Toroidal world support** for boundary wrapping
- **Comprehensive evaluation tools** including scalability testing
- **TensorBoard integration** for training visualization
- **Modular training utilities** for experiment reproducibility
- **Comprehensive testing suite** with pytest and benchmarks

## Quick Reference

### Main Training Command (TRPO - Hüttenrauch Parameters)

Train with TRPO algorithm using the parameters from the reference paper:

```bash
uv run python training/train_rendezvous.py \
  --algorithm trpo \
  --num-agents 10 \
  --world-size 100.0 \
  --comm-radius 75.0 \
  --obs-model local_basic \
  --total-timesteps 5000000 \
  --num-vec-envs 8 \
  --model-path models/huttenrauch_trpo.zip \
  --tensorboard-log training/tensorboard_logs
```

**TRPO Parameters (matching Hüttenrauch et al. 2019):**
- `n_steps=2048` (timesteps per batch)
- `target_kl=0.01` (max KL divergence)
- `learning_rate=1e-3` (value function learning rate)
- `cg_max_steps=10` (conjugate gradient iterations)
- `cg_damping=0.1` (damping coefficient)
- `gamma=0.99`, `gae_lambda=0.98`
- `n_critic_updates=5` (value function updates per policy update)

### Visual Evaluation Command

Evaluate a trained model with real-time visualization:

```bash
uv run python training/evaluate_rendezvous.py \
  --model-path models/huttenrauch_trpo.zip \
  --num-agents 10 \
  --world-size 100.0 \
  --comm-radius 75.0 \
  --obs-model local_basic \
  --render-mode human \
  --n-episodes 10
```

### Monitor Training Progress

Start TensorBoard to view training metrics in real-time:

```bash
tensorboard --logdir training/tensorboard_logs
# Open http://localhost:6006 in your browser
```

## Setup

### Requirements

- Python 3.11+
- CUDA-capable GPU (optional, for faster training)

### Installation

#### Option 1: Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd MARL-Swarm

# Sync dependencies (automatically creates virtual environment)
uv sync
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd MARL-Swarm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install in editable mode
pip install -e .
```

## Quick Start

### Training a Rendezvous Policy

**Train with default settings (4 agents):**

```bash
# Using uv
uv run python training/train_rendezvous.py

# Using standard python
python training/train_rendezvous.py
```

**Train with custom parameters:**

```bash
# Using uv
uv run python training/train_rendezvous.py \
  --num-agents 10 \
  --obs-model local_basic \
  --comm-radius 8.0 \
  --total-timesteps 300000 \
  --model-path models/my_model.zip

# Using standard python (with activated venv)
python training/train_rendezvous.py \
  --num-agents 10 \
  --obs-model local_basic \
  --comm-radius 8.0 \
  --total-timesteps 300000 \
  --model-path models/my_model.zip
```

**Train a scale-invariant policy (works with any number of agents):**

```bash
# Using uv
uv run python training/train_rendezvous.py \
  --num-agents 10 \
  --max-agents 50 \
  --total-timesteps 300000

# Using standard python
python training/train_rendezvous.py \
  --num-agents 10 \
  --max-agents 50 \
  --total-timesteps 300000
```

**Resume training from a checkpoint:**

```bash
# Using uv
uv run python training/train_rendezvous.py \
  --resume-from models/checkpoint.zip \
  --total-timesteps 500000 \
  --model-path models/continued.zip

# Using standard python
python training/train_rendezvous.py \
  --resume-from models/checkpoint.zip \
  --total-timesteps 500000 \
  --model-path models/continued.zip
```

**Train with TensorBoard logging:**

```bash
# Using uv
uv run python training/train_rendezvous.py \
  --num-agents 10 \
  --total-timesteps 300000 \
  --tensorboard-log training/tensorboard_logs

# Using standard python
python training/train_rendezvous.py \
  --num-agents 10 \
  --total-timesteps 300000 \
  --tensorboard-log training/tensorboard_logs

# View logs in TensorBoard
tensorboard --logdir training/tensorboard_logs
# Open http://localhost:6006 in your browser
```

### Evaluating a Trained Policy

**Evaluate with visualization:**

```bash
# Using uv
uv run python training/evaluate_rendezvous.py \
  --model-path models/my_model.zip \
  --num-agents 10 \
  --render-mode human \
  --n-episodes 5

# Using standard python
python training/evaluate_rendezvous.py \
  --model-path models/my_model.zip \
  --num-agents 10 \
  --render-mode human \
  --n-episodes 5
```

**Evaluate without rendering (for benchmarking):**

```bash
# Using uv
uv run python training/evaluate_rendezvous.py \
  --model-path models/my_model.zip \
  --num-agents 10 \
  --n-episodes 100

# Using standard python
python training/evaluate_rendezvous.py \
  --model-path models/my_model.zip \
  --num-agents 10 \
  --n-episodes 100
```

**Test scale-invariance (evaluate on multiple swarm sizes):**

```bash
# Using uv - run scalability experiments
uv run python evaluation/run_scalability_experiment.py \
  --train-size 4 \
  --test-sizes 4 8 16 32 50 \
  --total-timesteps 200000

# Using standard python
python evaluation/run_scalability_experiment.py \
  --train-size 4 \
  --test-sizes 4 8 16 32 50 \
  --total-timesteps 200000

# This will:
# 1. Train a policy on 4 agents (if not already trained)
# 2. Evaluate it on swarms of sizes 4, 8, 16, 32, 50
# 3. Generate plots and performance reports in evaluation/results/
```

### Using the Rendezvous Environment Programmatically

```python
from environments.rendezvous.rendezvous_env import RendezvousEnv

# Create environment
env = RendezvousEnv(
    num_agents=6,
    world_size=10.0,
    max_steps=100,
    obs_model="local_basic",
    comm_radius=8.0,
    render_mode="human",  # Set to None for no rendering
    fps=30
)

# Reset environment
obs, infos = env.reset()

# Run episode with random actions
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    env.render()  # Render if render_mode="human"

env.close()
```

## Training Parameters

### Common Arguments

- `--num-agents`: Number of agents in the environment (default: 4)
- `--world-size`: Side length of the square world (default: 10.0)
- `--max-steps`: Maximum steps per episode (default: 100)
- `--obs-model`: Observation model - `classic`, `global_basic`, `global_extended`, `local_basic`, `local_extended`, `local_comm` (default: `local_basic`)
- `--comm-radius`: Communication radius for local observation models (default: None)
- `--torus`: Enable toroidal world (wraps at boundaries)
- `--kinematics`: Agent kinematics - `single` or `double` (default: `single`)
- `--break-distance-threshold`: Early termination threshold for success (default: None)
- `--total-timesteps`: Training timesteps (default: 200,000)
- `--model-path`: Path to save the trained model (default: `rendezvous_model.zip`)
- `--resume-from`: Path to checkpoint to resume training from (default: None)
- `--tensorboard-log`: TensorBoard log directory for tracking training metrics (default: None)

### TensorBoard Logging

Enable TensorBoard logging to visualize training metrics in real-time:

```bash
# Using uv
uv run python training/train_rendezvous.py \
  --num-agents 10 \
  --total-timesteps 300000 \
  --tensorboard-log training/tensorboard_logs

# Using standard python
python training/train_rendezvous.py \
  --num-agents 10 \
  --total-timesteps 300000 \
  --tensorboard-log training/tensorboard_logs
```

View the logs in TensorBoard:

```bash
# Using uv
uv run tensorboard --logdir training/tensorboard_logs

# Using standard python (with activated venv)
tensorboard --logdir training/tensorboard_logs
```

Then open http://localhost:6006 in your browser to view training curves.

**Logged Metrics:**
- Episode reward mean
- Episode length mean
- Policy gradient loss
- Value function loss
- Entropy loss
- KL divergence
- Clip fraction
- Explained variance
- Learning rate

### Scale-Invariant Training

- `--max-agents`: Maximum number of agents the policy should support (enables scale-invariance)

When `--max-agents` is set, the policy can generalize to any number of agents up to that limit, even if trained with fewer agents.

## Project Structure

```
MARL-Swarm/
├── environments/
│   ├── base/                          # Base classes and utilities
│   │   ├── base_environment.py        # BaseEnv template pattern
│   │   ├── agent_handler.py           # NumPy-based agent state management
│   │   └── utils.py                   # Shared utilities
│   ├── rendezvous/                    # Rendezvous task (primary)
│   │   ├── rendezvous_env.py          # Main environment
│   │   ├── observations_vectorized.py # Optimized observation generation
│   │   ├── observations.py            # Legacy observation helpers
│   │   └── rendering.py               # Pygame rendering
│   ├── pursuit/                       # Pursuit-evasion task
│   │   ├── pursuit_evasion_env.py     # Pursuit-evasion environment
│   │   ├── observations.py            # Observation helpers
│   │   └── rendering.py               # Pygame rendering
│   └── tests/                         # Pytest test suite
│       ├── test_base/                 # Base class tests
│       ├── test_rendezvous/           # Rendezvous environment tests
│       ├── test_pursuit/              # Pursuit-evasion tests
│       ├── integration/               # PettingZoo API compliance
│       └── benchmarks/                # Performance benchmarks
├── policies/
│   └── meanEmbeddingExtractor.py      # Custom SB3 feature extractor
├── training/                          # Training and utilities
│   ├── train_rendezvous.py            # Main training script with CLI
│   ├── evaluate_rendezvous.py         # Evaluation script with visualization
│   ├── rendezvous_train_utils.py      # Reusable training utilities
│   ├── pursuit_evasion_train_utils.py # Pursuit-evasion training utilities
│   ├── configs/                       # Training configuration files
│   │   ├── embed_scaling.json         # Embedding dimension experiments
│   │   ├── depth_scaling.json         # Network depth experiments
│   │   └── ...                        # Additional experiment configs
│   └── __init__.py
├── evaluation/                        # Evaluation and analysis
│   ├── run_scalability_experiment.py  # Orchestrate scalability tests
│   ├── utils.py                       # Evaluation utilities
│   ├── plotting.py                    # Visualization and reporting
│   ├── results/                       # Experiment results and plots
│   └── __init__.py
├── models/                            # Saved trained model checkpoints
├── logs/                              # TensorBoard logs and training data
├── pyproject.toml                     # Project metadata and dependencies
├── requirements.txt                   # Runtime dependencies
├── README.md                          # This file
├── CLAUDE.md                          # Developer guidance
└── .gitignore                         # Git ignore rules
```

## Environments

### Rendezvous (Primary Task)

Agents learn to converge to a common location by minimizing pairwise distances. Key features:

- **Reward**: Global reward based on pairwise distances and action magnitude penalties
- **Success criterion**: Optional distance threshold for early termination
- **Observation models**: 6 different models from simple to advanced (see Observation Models below)
- **Kinematics**: Single or double integrator with configurable speed limits
- **Rendering**: Pygame-based visualization showing agent positions and communication

### Pursuit-Evasion (Implemented)

Multiple pursuers attempt to capture a single evader. Features:

- **Evader**: Scripted policy with configurable speed
- **Pursuers**: Trainable agents using the same policy with parameter sharing
- **Capture radius**: Configurable threshold for successful capture
- **Scaling**: Supports variable numbers of pursuers for scalability experiments
- **Observation models**: Same as rendezvous environment
- **Reward**: Collaborative reward for capturing the evader

## Observation Models

Both rendezvous and pursuit-evasion environments support multiple observation models:

1. **`classic`**: Own position, mean swarm position, velocities, orientation
2. **`global_basic`**: Wall distance/bearing + all neighbor distances/bearings
3. **`global_extended`**: Global basic + relative orientations and velocities
4. **`local_basic`**: Wall distance/bearing + neighbors within comm_radius (distance, bearing)
5. **`local_extended`**: Local basic + relative orientations
6. **`local_comm`**: Local extended + neighborhood size information

All non-classic models return **fixed-length observations** with:
- Deterministically ordered neighbors (sorted by distance)
- Padding with zeros for missing neighbors
- Binary masks indicating valid neighbors

This enables **scale-invariant policies** that generalize across different swarm sizes.

## Evaluation and Scalability Testing

The framework includes comprehensive evaluation tools:

### Scalability Experiments

Test how policies trained on small swarms generalize to larger swarms:

```bash
# Run scalability experiment
python evaluation/run_scalability_experiment.py \
  --train-size 4 \
  --test-sizes 4 8 16 32 50 \
  --total-timesteps 200000
```

This generates:
- Performance plots comparing different swarm sizes
- Success rate metrics
- Average reward and convergence speed analysis
- Comprehensive reports saved to `evaluation/results/`

### Evaluation Utilities

The `evaluation/` module provides:

- **`utils.py`**: Functions for loading models with normalization stats, evaluating on different swarm sizes, and computing metrics
- **`plotting.py`**: Visualization utilities for generating publication-quality plots and reports
- **`run_scalability_experiment.py`**: Orchestration script for systematic scalability testing

### Training Configuration Files

Pre-configured experiment setups in `training/configs/`:

- `embed_scaling.json`: Test different embedding dimensions
- `depth_scaling.json`: Test network depth variations
- `scalability_all_experiments.json`: Comprehensive scaling matrix
- Custom configs can be created following the same JSON format

## Testing

Run the test suite:

```bash
# Using uv
uv run pytest

# Using standard python
pytest
```

Run tests for a specific module:

```bash
# Using uv
uv run pytest environments/tests/test_rendezvous/

# Using standard python
pytest environments/tests/test_rendezvous/
```

Run with verbose output:

```bash
# Using uv
uv run pytest -v

# Using standard python
pytest -v
```

## Advanced Features

### Training Utilities Architecture

The `training/` module provides reusable utilities for experiment management:

**Main Components:**

1. **`rendezvous_train_utils.py`**: Core training pipeline
   - `wrap_env_for_sb3()`: Environment wrapping with SuperSuit
   - `make_policy_kwargs()`: Policy configuration with MeanEmbeddingExtractor
   - `setup_ppo_model()`: PPO initialization with proper device placement
   - `run_training_rendezvous()`: Complete training loop with checkpointing

2. **`train_rendezvous.py`**: CLI training script
   - Command-line argument parsing
   - Configuration management
   - Model and checkpoint saving
   - TensorBoard integration

3. **`evaluate_rendezvous.py`**: Evaluation script
   - Deterministic policy evaluation
   - Multiple rendering options
   - Automatic VecNormalize detection
   - Episode statistics collection

4. **`pursuit_evasion_train_utils.py`**: Pursuit-evasion training support

### Configuration Management

Training experiments can be configured via JSON files in `training/configs/`:

```json
{
  "num_agents": 4,
  "world_size": 10.0,
  "obs_model": "local_basic",
  "comm_radius": 8.0,
  "kinematics": "single",
  "embed_dim": 64,
  "total_timesteps": 200000
}
```

### Model Checkpointing and Resume

Models are saved with automatic VecNormalize statistics:

```bash
# Saves to: models/my_model.zip + models/my_model_vecnormalize.pkl
python training/train_rendezvous.py --model-path models/my_model.zip

# Resume from checkpoint
python training/train_rendezvous.py --resume-from models/my_model.zip \
  --total-timesteps 500000 --model-path models/continued.zip
```

### Performance Optimization

Key optimizations for training performance:

1. **Vectorized environment wrapping**: Multiple parallel environments for better GPU utilization
2. **NumPy-based agent state**: All agent properties stored as arrays for vectorized operations
3. **Optimized observation generation**: `observations_vectorized.py` uses efficient NumPy routines
4. **VecNormalize**: Observation and reward normalization for stable training

### Reproducibility

To ensure reproducible results:

1. Set random seeds via environment variables
2. Use deterministic evaluation (`--deterministic` flag in evaluate_rendezvous.py)
3. Save complete experiment configuration alongside models
4. VecNormalize statistics are automatically included with saved models

## Citation

If you use this code in your research, please cite:

```
[Add your thesis citation here once published]
```

## License

[Add license information]

## Contact

[Add your contact information]