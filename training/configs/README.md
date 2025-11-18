# Scalability Experiment Configurations

This directory contains JSON configuration files for systematic scalability experiments on the MARL-Swarm framework.

## Overview

All experiments are designed to test the **scale-invariance** and **generalization capabilities** of the MeanEmbedding architecture by varying:
1. **Network width** (embedding dimension: 8 to 1024)
2. **Network depth** (phi_layers: 1 to 8)
3. **Training swarm size** (4 to 200 agents)
4. **Testing swarm size** (4 to 1000 agents)

**Key setting**: All experiments use `max_agents: 1000` to enable scale-invariant observations.

## Experiment Types

### Single-Variable Experiments
Test one parameter at a time while keeping others fixed (e.g., vary embedding dimension with fixed training size).

### Matrix Experiments
Systematically test combinations of **network architecture × training swarm size** to understand:
- Does training on larger swarms compensate for smaller networks?
- Do larger networks trained on small swarms outperform smaller networks trained on large swarms?
- What's the optimal (network size, training size) combination for generalization?

## Configuration Files Summary

| Config File | Experiments | Type | Description |
|-------------|-------------|------|-------------|
| `matrix_quick.json` | 15 | Matrix | Quick validation matrix (3-4 hrs) |
| `embed_scaling.json` | 5 | Single-var | Embedding dimension sweep |
| `depth_scaling.json` | 4 | Single-var | Network depth sweep |
| `transferability_small_to_large.json` | 3 | Single-var | Train small → test huge |
| `transferability_large_to_small.json` | 3 | Single-var | Train large → test varied |
| `edge_cases.json` | 7 | Single-var | Extreme architectures |
| `scalability_all_experiments.json` | 20 | Combined | All single-variable experiments |
| `matrix_embed_x_swarmsize.json` | 20 | Matrix | 5 embed_dims × 4 train_sizes |
| `matrix_depth_x_swarmsize.json` | 16 | Matrix | 4 depths × 4 train_sizes |
| **Total** | **71** | — | Full experimental suite |

---

## Configuration Files

### 1. `scalability_all_experiments.json` (Master Config)
**20 experiments total** - Run all experiments at once:
```bash
python evaluation/run_scalability_experiment.py --config training/configs/scalability_all_experiments.json
```

Includes:
- 5 embedding dimension experiments (16, 32, 64, 128, 256)
- 4 network depth experiments (1, 2, 3, 4 layers)
- 5 transferability experiments (small→large and large→small)
- 6 edge case experiments (extreme architectures)

---

### 2. `embed_scaling.json` - Embedding Dimension Scaling
**5 experiments** - Test how network width affects generalization:

| Experiment | embed_dim | phi_layers | train_size | test_sizes |
|------------|-----------|------------|------------|------------|
| embed_16   | 16        | 1          | 4          | [4, 8, 16, 32, 50, 100] |
| embed_32   | 32        | 1          | 4          | [4, 8, 16, 32, 50, 100] |
| embed_64   | 64        | 1          | 4          | [4, 8, 16, 32, 50, 100] |
| embed_128  | 128       | 1          | 4          | [4, 8, 16, 32, 50, 100] |
| embed_256  | 256       | 1          | 4          | [4, 8, 16, 32, 50, 100] |

**Hypothesis**: Larger embeddings → better neighbor feature representation → improved generalization to large swarms

**Run**:
```bash
python evaluation/run_scalability_experiment.py --config training/configs/embed_scaling.json
```

---

### 3. `depth_scaling.json` - Network Depth Scaling
**4 experiments** - Test how network depth affects feature learning:

| Experiment | embed_dim | phi_layers | train_size | test_sizes |
|------------|-----------|------------|------------|------------|
| depth_1    | 64        | 1          | 4          | [4, 8, 16, 32, 50, 100] |
| depth_2    | 64        | 2          | 4          | [4, 8, 16, 32, 50, 100] |
| depth_3    | 64        | 3          | 4          | [4, 8, 16, 32, 50, 100] |
| depth_4    | 64        | 4          | 4          | [4, 8, 16, 32, 50, 100] |

**Hypothesis**: Deeper φ networks → richer neighbor representations → better scalability

**Run**:
```bash
python evaluation/run_scalability_experiment.py --config training/configs/depth_scaling.json
```

---

### 4. `transferability_small_to_large.json` - Small→Large Transfer
**3 experiments** - Test generalization from small training swarms to very large test swarms:

| Experiment | embed_dim | phi_layers | train_size | test_sizes |
|------------|-----------|------------|------------|------------|
| transfer_4_to_extreme  | 64 | 2 | 4  | [4, 10, 20, 40, 80, 150, 200, 300, 500, 800] |
| transfer_8_to_extreme  | 64 | 2 | 8  | [8, 16, 32, 64, 128, 200, 300, 500, 800] |
| transfer_16_to_extreme | 64 | 2 | 16 | [16, 32, 64, 100, 150, 250, 400, 600, 1000] |

**Hypothesis**: Scale-invariant architecture enables policies trained on 4-16 agents to generalize to 100-1000 agents

**Run**:
```bash
python evaluation/run_scalability_experiment.py --config training/configs/transferability_small_to_large.json
```

**Expected observations**:
- Performance degradation as swarm size increases
- Potential breakdown point where coordination fails
- Testing limits of scale-invariance

---

### 5. `transferability_large_to_small.json` - Large→Small Transfer
**3 experiments** - Test reverse transfer (train on large, test on small):

| Experiment | embed_dim | phi_layers | train_size | test_sizes |
|------------|-----------|------------|------------|------------|
| transfer_50_to_varied  | 64 | 2 | 50  | [4, 8, 12, 16, 24, 32, 40, 50, 75, 100, 150, 200] |
| transfer_100_to_varied | 64 | 2 | 100 | [4, 8, 16, 32, 50, 75, 100, 150, 200, 300] |
| transfer_200_to_varied | 64 | 2 | 200 | [10, 25, 50, 100, 150, 200, 300, 400, 500] |

**Hypothesis**: Training on large swarms may:
- Provide more diverse training signal
- Lead to more robust policies
- Or overfit to large-swarm dynamics and fail on small swarms

**Run**:
```bash
python evaluation/run_scalability_experiment.py --config training/configs/transferability_large_to_small.json
```

---

### 6. `edge_cases.json` - Extreme Network Architectures
**7 experiments** - Test edge cases to find failure modes:

| Experiment | embed_dim | phi_layers | Notes |
|------------|-----------|------------|-------|
| edge_tiny_net    | 8    | 1 | Minimal capacity - expect underfitting |
| edge_huge_net    | 512  | 1 | Very wide - potential overfitting |
| edge_very_deep   | 64   | 6 | Deep but not wide |
| edge_tiny_deep   | 16   | 4 | Narrow + deep - gradient issues? |
| edge_huge_deep   | 256  | 4 | Large capacity - best performance? |
| edge_ultra_wide  | 1024 | 1 | Extreme width - diminishing returns? |
| edge_ultra_deep  | 64   | 8 | Extreme depth - training instability? |

**Run**:
```bash
python evaluation/run_scalability_experiment.py --config training/configs/edge_cases.json
```

**Expected edge case behaviors**:
- **Tiny nets** (8-16 dim): Poor performance, limited capacity
- **Huge nets** (512-1024 dim): Overfitting on small training swarms, slower training
- **Very deep** (6-8 layers): Potential gradient issues, longer training time
- **Optimal zone**: Likely 64-128 dim with 2-4 layers

---

### 7. `matrix_embed_x_swarmsize.json` - Embedding Dimension × Training Swarm Size Matrix
**20 experiments** - Systematic matrix analysis of network width vs. training size:

| embed_dim | train_size=4 | train_size=8 | train_size=16 | train_size=32 |
|-----------|--------------|--------------|---------------|---------------|
| 16        | ✓            | ✓            | ✓             | ✓             |
| 32        | ✓            | ✓            | ✓             | ✓             |
| 64        | ✓            | ✓            | ✓             | ✓             |
| 128       | ✓            | ✓            | ✓             | ✓             |
| 256       | ✓            | ✓            | ✓             | ✓             |

All use phi_layers=1, test on [4, 8, 16, 32, 50, 100]

**Research Questions**:
- Does a larger network (256 dim) trained on 4 agents outperform a smaller network (32 dim) trained on 16 agents?
- Is there a "sweet spot" combination?
- How does the generalization gap change across the matrix?

**Run**:
```bash
python evaluation/run_scalability_experiment.py --config training/configs/matrix_embed_x_swarmsize.json
```

---

### 8. `matrix_depth_x_swarmsize.json` - Network Depth × Training Swarm Size Matrix
**16 experiments** - Systematic matrix analysis of network depth vs. training size:

| phi_layers | train_size=4 | train_size=8 | train_size=16 | train_size=32 |
|------------|--------------|--------------|---------------|---------------|
| 1          | ✓            | ✓            | ✓             | ✓             |
| 2          | ✓            | ✓            | ✓             | ✓             |
| 3          | ✓            | ✓            | ✓             | ✓             |
| 4          | ✓            | ✓            | ✓             | ✓             |

All use embed_dim=64, test on [4, 8, 16, 32, 50, 100]

**Research Questions**:
- Does depth help more when training on small or large swarms?
- Can deep networks trained on small swarms generalize better than shallow networks trained on large swarms?
- How does depth affect sample efficiency across different training sizes?

**Run**:
```bash
python evaluation/run_scalability_experiment.py --config training/configs/matrix_depth_x_swarmsize.json
```

---

### 9. `matrix_quick.json` - Quick Matrix Experiments (Fast Iteration)
**15 experiments** - Reduced matrix for rapid prototyping:

- **embed_dims**: [32, 64, 128]
- **phi_layers**: [1, 2, 3]
- **train_sizes**: [4, 8, 16]
- **test_sizes**: [4, 8, 16, 32, 50]
- **total_timesteps**: 300,000 (faster training)
- **n_eval_episodes**: 50 (faster evaluation)

This configuration is designed for:
- Quick validation of the experimental pipeline
- Preliminary results before running full experiments
- Debugging and parameter tuning

**Run**:
```bash
python evaluation/run_scalability_experiment.py --config training/configs/matrix_quick.json
```

**Estimated time**: ~3-4 hours (vs. ~8-12 hours for full experiments)

---

## Environment Configuration (Common)

All experiments use identical environment settings:
```json
{
  "world_size": 10.0,
  "max_steps": 100,
  "obs_model": "local_basic",
  "comm_radius": 8.0,
  "kinematics": "single",
  "torus": false,
  "max_agents": 1000
}
```

**Critical**: `max_agents: 1000` ensures observations are padded/masked to support up to 1000 agents, enabling scale-invariant policies.

---

## Training Configuration (Common)

All experiments use:
```json
{
  "total_timesteps": 500000,
  "ppo_params": {"learning_rate": 0.0003},
  "n_envs": 1,
  "normalize": true
}
```

---

## Evaluation Configuration

All experiments evaluate with:
- **n_eval_episodes**: 100 (per swarm size)
- **deterministic**: true (use policy without exploration noise)

---

## Usage Examples

### Quick start (recommended for initial testing):
```bash
# Run quick matrix for preliminary results (~3-4 hours)
python evaluation/run_scalability_experiment.py --config training/configs/matrix_quick.json
```

### Run single-variable experiment categories:
```bash
# Embedding scaling experiments (5 experiments)
python evaluation/run_scalability_experiment.py --config training/configs/embed_scaling.json

# Depth scaling experiments (4 experiments)
python evaluation/run_scalability_experiment.py --config training/configs/depth_scaling.json

# Transferability experiments
python evaluation/run_scalability_experiment.py --config training/configs/transferability_small_to_large.json  # 3 experiments
python evaluation/run_scalability_experiment.py --config training/configs/transferability_large_to_small.json  # 3 experiments

# Edge cases (7 experiments)
python evaluation/run_scalability_experiment.py --config training/configs/edge_cases.json
```

### Run matrix experiment categories:
```bash
# Embedding × Swarm Size matrix (20 experiments)
python evaluation/run_scalability_experiment.py --config training/configs/matrix_embed_x_swarmsize.json

# Depth × Swarm Size matrix (16 experiments)
python evaluation/run_scalability_experiment.py --config training/configs/matrix_depth_x_swarmsize.json
```

### Run all single-variable experiments:
```bash
python evaluation/run_scalability_experiment.py --config training/configs/scalability_all_experiments.json
```

### Run ALL experiments (single-variable + matrices):
```bash
# Option 1: Run sequentially (one config at a time)
python evaluation/run_scalability_experiment.py --config training/configs/scalability_all_experiments.json
python evaluation/run_scalability_experiment.py --config training/configs/matrix_embed_x_swarmsize.json
python evaluation/run_scalability_experiment.py --config training/configs/matrix_depth_x_swarmsize.json

# Total: 20 + 20 + 16 = 56 experiments
```

### Force retrain all models:
Modify any config file and set `"force_retrain": true` for experiments you want to retrain.

---

## Expected Output

For each experiment, the script will:

1. **Train model** (or load existing) → `evaluation/results/<exp_name>/<exp_name>_model.zip`
2. **Evaluate on test sizes** → Episode statistics for each swarm size
3. **Generate plots** → `evaluation/results/<exp_name>/plots/`
   - Mean reward vs swarm size
   - Success rate vs swarm size
   - Final max distance vs swarm size
   - Episode length vs swarm size
4. **Save configuration** → `evaluation/results/<exp_name>/<exp_name>_config.json`
5. **Print summary table** → Console output with key metrics

### Comparison plots
When running multiple experiments, comparison plots are saved to:
`evaluation/results/comparisons/comparison_<metric>.png`

---

## Analysis Guidelines

### Key Metrics to Compare:

1. **Generalization Gap**:
   - Performance drop from `train_size` to larger `test_sizes`
   - Smaller gap = better generalization

2. **Success Rate**:
   - Percentage of episodes meeting rendezvous criterion
   - Should ideally remain high across all test sizes

3. **Mean Final Max Distance**:
   - Lower = better rendezvous
   - Track how this scales with swarm size

4. **Sample Efficiency**:
   - Training time vs. final performance
   - Larger networks may need more timesteps

### Questions to Answer:

**Single-Variable Experiments**:
1. **Embedding dimension**: What's the optimal width? Does 256 outperform 64 on large swarms?
2. **Network depth**: Does deeper feature extraction help? Or cause overfitting?
3. **Transferability**: Can we train on 4 agents and deploy on 1000?
4. **Reverse transfer**: Does training on large swarms help small swarms?
5. **Edge cases**: What's the minimum viable network? What's the point of diminishing returns?

**Matrix Experiments**:
6. **Architecture vs. Training Size**: Does a large network trained on few agents beat a small network trained on many agents?
7. **Interaction Effects**: Are there synergies between network capacity and training swarm size?
8. **Optimal Combination**: What (embed_dim, train_size) pair gives best generalization to large swarms?
9. **Depth vs. Data**: Does depth compensate for smaller training swarms?

---

## Notes

- All experiments use `max_agents: 1000` for scale-invariance
- Training timesteps: 500,000 (may need adjustment for larger networks)
- Evaluation episodes: 100 per size (statistically significant)
- Models are saved with VecNormalize statistics for reproducibility
- Set `force_retrain: true` to retrain existing models

---

## Computational Considerations

**Estimated time per experiment** (on typical hardware):
- Training (500k steps): 10-30 minutes (depends on network size)
- Training (300k steps, quick matrix): 6-18 minutes
- Evaluation (100 episodes × 6 sizes): 5-10 minutes
- Evaluation (50 episodes × 5 sizes, quick matrix): 3-5 minutes

**Total time estimates**:
- `matrix_quick.json` (15 experiments): ~3-4 hours
- `scalability_all_experiments.json` (20 experiments): ~8-12 hours
- `matrix_embed_x_swarmsize.json` (20 experiments): ~8-12 hours
- `matrix_depth_x_swarmsize.json` (16 experiments): ~6-10 hours
- **All experiments combined (71 total)**: ~30-40 hours

**Recommended workflow for thesis**:

### Phase 1: Quick Validation (~4 hours)
```bash
python evaluation/run_scalability_experiment.py --config training/configs/matrix_quick.json
```
- Validates experimental pipeline
- Provides preliminary insights
- Identifies potential issues

### Phase 2: Single-Variable Analysis (~8-12 hours)
```bash
# Run in this order for iterative insights
python evaluation/run_scalability_experiment.py --config training/configs/embed_scaling.json
python evaluation/run_scalability_experiment.py --config training/configs/depth_scaling.json
python evaluation/run_scalability_experiment.py --config training/configs/transferability_small_to_large.json
python evaluation/run_scalability_experiment.py --config training/configs/edge_cases.json
```

### Phase 3: Matrix Analysis (~14-22 hours)
```bash
# Based on insights from Phase 2, run relevant matrix experiments
python evaluation/run_scalability_experiment.py --config training/configs/matrix_embed_x_swarmsize.json
python evaluation/run_scalability_experiment.py --config training/configs/matrix_depth_x_swarmsize.json
```

### Phase 4: Final Analysis
- Generate comparison plots across all experiments
- Identify optimal configurations
- Write thesis chapter with comprehensive results

---

## Customization

To modify experiments, edit the JSON files directly. Key parameters:

```json
{
  "train_config": {
    "total_timesteps": 500000,        // Increase for better training
    "embed_config": {
      "embed_dim": 64,                // Network width
      "phi_layers": 2                 // Network depth
    },
    "ppo_params": {
      "learning_rate": 0.0003         // Adjust if training unstable
    }
  },
  "env_config": {
    "num_agents": 4,                  // Training swarm size
    "max_agents": 1000,               // KEEP HIGH for scale-invariance
    "comm_radius": 8.0,               // Local observation radius
    "world_size": 10.0                // Environment size
  },
  "test_sizes": [4, 8, 16, 32, 50, 100],  // Evaluation swarm sizes
  "n_eval_episodes": 100,             // Episodes per size
  "force_retrain": false              // Set true to retrain
}
```

---

## Troubleshooting

**Training fails / unstable**:
- Reduce learning_rate (try 1e-4)
- Reduce network size for very deep networks
- Increase total_timesteps

**Evaluation too slow**:
- Reduce n_eval_episodes (50 instead of 100)
- Reduce test_sizes (remove extreme values)

**Out of memory**:
- Reduce embed_dim or phi_layers
- Train with smaller max_agents (but loses scale-invariance!)

---

## Citation

If you use these configurations in your thesis or publications, please reference the MARL-Swarm framework and the MeanEmbedding architecture.
