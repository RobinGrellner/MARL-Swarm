#!/usr/bin/env python3
"""
Rendezvous Embedding Scaling Analysis
======================================

Comprehensive analysis of embedding dimension scaling experiments for the rendezvous task.

Experiment Matrix:
  - Agent Counts: [4, 16, 50, 100]
  - Embedding Dimensions: [4, 8, 32, 64 (Hüttenrauch baseline), 128]
  - Seeds: [0] (core) + [123] (extensions)
  - Total: 20 core configurations

Usage:
    python analyze_rendezvous_embedding.py

Outputs:
    - Console summary with statistics
    - Visualization PNG files
    - CSV export with detailed metrics
    - Comparison to Hüttenrauch baseline

References:
    Hüttenrauch et al. (2019). Deep Reinforcement Learning for Swarm Systems.
    arXiv:1807.06613
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import warnings
import sys

warnings.filterwarnings('ignore')

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available. Skipping visualizations.", file=sys.stderr)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Configuration
BASE_DIR = Path("logs/rendezvous_embedding")
OUTPUT_DIR = Path("results/rendezvous_scaling")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    'baseline': '#FF6B6B',  # Red
    'better': '#51CF66',     # Green
    'worse': '#FFD93D',      # Yellow
    'neutral': '#6C63FF'     # Blue
}

def parse_experiment_name(exp_name):
    """Parse experiment directory name to extract parameters.

    Format: num_agents{N}_embed_dim{D}_seed{S}
    But the underscore split gives: ['num', 'agents{N}', 'embed', 'dim{D}', 'seed{S}']
    """
    params = {}
    parts = exp_name.split('_')

    for i, part in enumerate(parts):
        if part == 'num' and i+1 < len(parts) and parts[i+1].startswith('agents'):
            # Extract number from 'agents100'
            num_str = parts[i+1][6:]  # Remove 'agents' prefix
            if num_str.isdigit():
                params['num_agents'] = int(num_str)

        elif part == 'embed' and i+1 < len(parts) and parts[i+1].startswith('dim'):
            # Extract number from 'dim128'
            dim_str = parts[i+1][3:]  # Remove 'dim' prefix
            if dim_str.isdigit():
                params['embed_dim'] = int(dim_str)

        elif part.startswith('seed'):
            # Extract number from 'seed0'
            seed_str = part[4:]  # Remove 'seed' prefix
            if seed_str.isdigit():
                params['seed'] = int(seed_str)

    return params


def load_event_file(event_path):
    """Load TensorBoard event file and extract metrics."""
    try:
        ea = EventAccumulator(str(event_path))
        ea.Reload()

        metrics = {}
        scalar_keys = ea.Tags()['scalars']

        for key in scalar_keys:
            try:
                events = ea.Scalars(key)
                values = np.array([e.value for e in events])
                steps = np.array([e.step for e in events])
                metrics[key] = {'steps': steps, 'values': values, 'events': events}
            except:
                pass

        return metrics
    except Exception as e:
        print(f"Error loading {event_path}: {e}", file=sys.stderr)
        return None


def extract_summary_stats(metrics_dict, key_patterns):
    """Extract summary statistics from metrics."""
    stats = {}

    if metrics_dict is None:
        return stats

    for pattern in key_patterns:
        matching_keys = [k for k in metrics_dict.keys() if pattern.lower() in k.lower()]

        if matching_keys:
            key = matching_keys[0]
            values = metrics_dict[key]['values']
            steps = metrics_dict[key]['steps']

            if len(values) > 0:
                stats[f'{pattern}_final'] = float(values[-1])
                stats[f'{pattern}_max'] = float(np.max(values))
                stats[f'{pattern}_min'] = float(np.min(values))
                stats[f'{pattern}_mean'] = float(np.mean(values))
                stats[f'{pattern}_std'] = float(np.std(values))
                stats[f'{pattern}_n_steps'] = len(values)

                # Convergence iteration
                if np.min(values) != np.max(values):
                    target = np.min(values) + 0.8 * (np.max(values) - np.min(values))
                    convergence_idx = np.argmax(values >= target)
                    if convergence_idx > 0:
                        stats[f'{pattern}_converge_iter'] = int(steps[convergence_idx])

    return stats


def load_all_experiments():
    """Load all experiment data from TensorBoard logs."""
    results = []
    errors = []

    if not BASE_DIR.exists():
        print(f"ERROR: Base directory not found: {BASE_DIR}")
        return pd.DataFrame(), []

    exp_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(exp_dirs)} experiment directories")

    for exp_dir in exp_dirs:
        try:
            params = parse_experiment_name(exp_dir.name)
            trpo_dirs = list(exp_dir.glob("TRPO_*"))

            if not trpo_dirs:
                errors.append(f"{exp_dir.name}: No TRPO directory")
                continue

            event_files = list(trpo_dirs[0].glob("events.out.tfevents*"))
            if not event_files:
                errors.append(f"{exp_dir.name}: No event file")
                continue

            metrics = load_event_file(event_files[0])
            if metrics is None:
                errors.append(f"{exp_dir.name}: Failed to load event file")
                continue

            stats = extract_summary_stats(metrics, [
                'rollout/ep_rew',
                'rollout/ep_len',
                'train/policy_loss',
                'train/value_loss',
                'rollout/success'
            ])

            record = {**params, **stats, 'exp_name': exp_dir.name}
            results.append(record)

        except Exception as e:
            errors.append(f"{exp_dir.name}: {str(e)}")

    return pd.DataFrame(results), errors


def print_summary_table(df, reward_col):
    """Print comprehensive summary table."""
    print("\n" + "="*100)
    print("EXPERIMENT SUMMARY TABLE (20 Core Configurations)")
    print("="*100)
    print()

    summary = df[['exp_name', 'num_agents', 'embed_dim', reward_col]].copy()
    summary.columns = ['Experiment', 'Agents', 'Embedding Dim', 'Final Reward']
    summary = summary.sort_values('Final Reward', ascending=False)
    summary['Rank'] = range(1, len(summary) + 1)
    summary = summary[['Rank', 'Agents', 'Embedding Dim', 'Final Reward', 'Experiment']]

    for idx, row in summary.iterrows():
        print(f"{int(row['Rank']):2d}. N={int(row['Agents']):3d}, d={int(row['Embedding Dim']):3d} | "
              f"Reward: {row['Final Reward']:8.3f} | {row['Experiment']}")


def print_key_findings(df, reward_col):
    """Print key findings and insights."""
    print("\n" + "="*100)
    print("KEY FINDINGS")
    print("="*100)
    print()

    # Best and worst
    best_idx = df[reward_col].idxmax()
    worst_idx = df[reward_col].idxmin()
    best = df.loc[best_idx]
    worst = df.loc[worst_idx]

    print(f"1. BEST CONFIGURATION")
    print(f"   N={int(best['num_agents'])}, d={int(best['embed_dim'])}: {best[reward_col]:.3f}")
    print()

    print(f"2. WORST CONFIGURATION")
    print(f"   N={int(worst['num_agents'])}, d={int(worst['embed_dim'])}: {worst[reward_col]:.3f}")
    print()

    # Baseline analysis
    baseline_df = df[df['embed_dim'] == 64]
    print(f"3. HÜTTENRAUCH BASELINE (d=64)")
    print(f"   Mean Reward: {baseline_df[reward_col].mean():.3f} ± {baseline_df[reward_col].std():.3f}")
    print(f"   Range: [{baseline_df[reward_col].min():.3f}, {baseline_df[reward_col].max():.3f}]")
    print()

    # Embedding dimension analysis
    print(f"4. PERFORMANCE BY EMBEDDING DIMENSION")
    embed_stats = df.groupby('embed_dim')[reward_col].agg(['mean', 'std', 'min', 'max'])
    for d in embed_stats.index:
        row = embed_stats.loc[d]
        marker = " [BASELINE]" if d == 64 else ""
        print(f"   d={d:3d}: mean={row['mean']:.3f}, std={row['std']:.3f}, "
              f"range=[{row['min']:.3f}, {row['max']:.3f}]{marker}")
    print()

    # Optimal per agent count
    print(f"5. OPTIMAL EMBEDDING FOR EACH AGENT COUNT")
    for n_agents in sorted(df['num_agents'].unique()):
        subset = df[df['num_agents'] == n_agents]
        optimal = subset.loc[subset[reward_col].idxmax()]
        baseline = subset[subset['embed_dim'] == 64]

        if not baseline.empty:
            baseline_reward = baseline[reward_col].values[0]
            pct_diff = (optimal[reward_col] - baseline_reward) / abs(baseline_reward) * 100
            print(f"   N={int(n_agents):3d}: d={int(optimal['embed_dim']):3d} "
                  f"(reward={optimal[reward_col]:7.3f}, {pct_diff:+.1f}% vs d=64)")


def plot_heatmaps(df, reward_col):
    """Create performance heatmaps."""
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        print("  Skipping heatmaps (matplotlib/seaborn not available)", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pivot table
    heatmap_data = df.pivot_table(
        values=reward_col,
        index='num_agents',
        columns='embed_dim',
        aggfunc='mean'
    )

    # Absolute rewards
    ax = axes[0]
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
                cbar_kws={'label': 'Final Reward'})
    ax.set_title('Performance Heatmap\n(Agent Count × Embedding Dimension)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Embedding Dimension', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Agents', fontsize=11, fontweight='bold')

    # Normalized
    ax = axes[1]
    heatmap_norm = heatmap_data.div(heatmap_data.max(axis=1), axis=0)
    sns.heatmap(heatmap_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Normalized Performance'})
    ax.set_title('Normalized Performance (Best=1.0)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Embedding Dimension', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Agents', fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_file = OUTPUT_DIR / "01_heatmaps.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_dimensional_reduction(df, reward_col):
    """Plot embedding dimension scaling analysis."""
    if not HAS_MATPLOTLIB:
        print("  Skipping dimensional reduction plot (matplotlib not available)", file=sys.stderr)
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    agent_counts = sorted(df['num_agents'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(agent_counts)))

    for idx, n_agents in enumerate(agent_counts):
        ax = axes[idx // 2, idx % 2]
        subset = df[df['num_agents'] == n_agents].sort_values('embed_dim')

        ax.plot(subset['embed_dim'], subset[reward_col], 'o-',
               linewidth=2.5, markersize=8, color=colors[idx])

        # Highlight baseline
        baseline = subset[subset['embed_dim'] == 64]
        if not baseline.empty:
            baseline_reward = baseline[reward_col].values[0]
            ax.scatter(64, baseline_reward, s=300, marker='s', color='red',
                      edgecolor='darkred', linewidth=2, label='Baseline (d=64)', zorder=5)

        ax.set_xlabel('Embedding Dimension', fontsize=11, fontweight='bold')
        ax.set_ylabel('Final Reward', fontsize=11, fontweight='bold')
        ax.set_title(f'Performance Scaling: {n_agents} Agents', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_xticks([4, 8, 32, 64, 128])

    plt.tight_layout()
    output_file = OUTPUT_DIR / "02_dimensional_reduction.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_scalability(df, reward_col):
    """Plot swarm size scalability analysis."""
    if not HAS_MATPLOTLIB:
        print("  Skipping scalability plot (matplotlib not available)", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    embed_dims = sorted(df['embed_dim'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(embed_dims)))

    # Plot 1: Separate lines for each embedding dimension
    ax = axes[0]
    for idx, d in enumerate(embed_dims):
        subset = df[df['embed_dim'] == d].sort_values('num_agents')
        ax.plot(subset['num_agents'], subset[reward_col], 'o-',
               linewidth=2.5, markersize=8, label=f'd={d}', color=colors[idx])

    ax.set_xlabel('Number of Agents', fontsize=11, fontweight='bold')
    ax.set_ylabel('Final Reward', fontsize=11, fontweight='bold')
    ax.set_title('Scalability: Reward vs Swarm Size', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # Plot 2: Delta from baseline
    ax = axes[1]
    baseline_df = df[df['embed_dim'] == 64].copy()
    baseline_dict = dict(zip(baseline_df['num_agents'], baseline_df[reward_col]))

    for idx, d in enumerate(embed_dims):
        subset = df[df['embed_dim'] == d].sort_values('num_agents')
        deltas = []

        for _, row in subset.iterrows():
            baseline = baseline_dict.get(row['num_agents'])
            if baseline is not None:
                delta = (row[reward_col] - baseline) / abs(baseline) * 100 if baseline != 0 else 0
                deltas.append(delta)

        if deltas:
            ax.plot(subset['num_agents'], deltas, 'o-',
                   linewidth=2.5, markersize=8, label=f'd={d}', color=colors[idx])

    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Number of Agents', fontsize=11, fontweight='bold')
    ax.set_ylabel('Performance Delta (% vs d=64)', fontsize=11, fontweight='bold')
    ax.set_title('Relative Performance vs Baseline', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "03_scalability.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_saturation(df, reward_col):
    """Plot embedding dimension saturation analysis."""
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        print("  Skipping saturation plot (matplotlib/seaborn not available)", file=sys.stderr)
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    agent_counts = sorted(df['num_agents'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(agent_counts)))

    # Plot 1: Improvement range
    ax = axes[0, 0]
    improvements = []
    for n_agents in agent_counts:
        subset = df[df['num_agents'] == n_agents]
        improvement = subset[reward_col].max() - subset[reward_col].min()
        improvements.append(improvement)

    ax.bar(range(len(agent_counts)), improvements, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(agent_counts)))
    ax.set_xticklabels([f'N={n}' for n in agent_counts])
    ax.set_ylabel('Reward Range', fontsize=11, fontweight='bold')
    ax.set_title('Impact of Embedding Dimension', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Efficiency
    ax = axes[0, 1]
    for idx, n_agents in enumerate(agent_counts):
        subset = df[df['num_agents'] == n_agents].sort_values('embed_dim')
        efficiency = subset[reward_col] / (subset['embed_dim'] / 64.0)
        ax.plot(subset['embed_dim'], efficiency, 'o-', linewidth=2, markersize=8,
               color=colors[idx], label=f'N={n_agents}')

    ax.set_xlabel('Embedding Dimension', fontsize=11, fontweight='bold')
    ax.set_ylabel('Efficiency (Reward per Norm. Dim)', fontsize=11, fontweight='bold')
    ax.set_title('Embedding Efficiency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_xticks([4, 8, 32, 64, 128])

    # Plot 3: Relative performance
    ax = axes[1, 0]
    ratios = []
    for n_agents in agent_counts:
        d4 = df[(df['num_agents'] == n_agents) & (df['embed_dim'] == 4)][reward_col].values
        d64 = df[(df['num_agents'] == n_agents) & (df['embed_dim'] == 64)][reward_col].values
        d128 = df[(df['num_agents'] == n_agents) & (df['embed_dim'] == 128)][reward_col].values

        if len(d4) > 0 and len(d64) > 0 and len(d128) > 0:
            ratios.append({'agent_count': n_agents, 'd4/d64': d4[0] / d64[0], 'd128/d64': d128[0] / d64[0]})

    if ratios:
        ratio_df = pd.DataFrame(ratios)
        x = np.arange(len(ratio_df))
        width = 0.35

        ax.bar(x - width/2, ratio_df['d4/d64'], width, label='d=4 / d=64', alpha=0.8, color='coral')
        ax.bar(x + width/2, ratio_df['d128/d64'], width, label='d=128 / d=64', alpha=0.8, color='lightgreen')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')

        ax.set_xticks(x)
        ax.set_xticklabels([f'N={n}' for n in ratio_df['agent_count']])
        ax.set_ylabel('Performance Ratio', fontsize=11, fontweight='bold')
        ax.set_title('Relative vs Baseline', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = []
    for n_agents in agent_counts:
        subset = df[df['num_agents'] == n_agents]
        table_data.append([
            f'N={n_agents}',
            f"{subset[reward_col].min():.2f}",
            f"{subset[reward_col].max():.2f}",
            f"{subset[reward_col].mean():.2f}"
        ])

    table = ax.table(cellText=table_data,
                    colLabels=['Agents', 'Min', 'Max', 'Mean'],
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "04_saturation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main analysis pipeline."""
    print("="*100)
    print("RENDEZVOUS EMBEDDING SCALING ANALYSIS")
    print("="*100)
    print()

    # Load experiments
    print("Loading TensorBoard event files...")
    df, errors = load_all_experiments()

    if len(df) == 0:
        print("ERROR: No experiments loaded!")
        if errors:
            print("\nErrors:")
            for err in errors[:10]:
                print(f"  - {err}")
        return 1

    print(f"Successfully loaded {len(df)} experiments\n")

    if errors:
        print(f"Warnings ({len(errors)} errors):")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors)-5} more")
        print()

    # Identify reward column
    reward_cols = [c for c in df.columns if 'ep_rew' in c and 'final' in c]
    if not reward_cols:
        reward_cols = [c for c in df.columns if 'ep_rew' in c]

    if not reward_cols:
        print("ERROR: No reward column found!")
        return 1

    reward_col = reward_cols[0]
    print(f"Using reward metric: {reward_col}\n")

    # Print summary
    print_summary_table(df, reward_col)
    print_key_findings(df, reward_col)

    # Create visualizations
    print("\n" + "="*100)
    print("CREATING VISUALIZATIONS")
    print("="*100)
    print()

    plot_heatmaps(df, reward_col)
    plot_dimensional_reduction(df, reward_col)
    plot_scalability(df, reward_col)
    plot_saturation(df, reward_col)

    # Export results
    print("\n" + "="*100)
    print("EXPORTING RESULTS")
    print("="*100)
    print()

    output_csv = OUTPUT_DIR / "rendezvous_scaling_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"Exported: {output_csv}")

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"\nResults saved to: {OUTPUT_DIR}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
