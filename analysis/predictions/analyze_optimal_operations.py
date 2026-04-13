#!/usr/bin/env python3
"""
Analyze optimal module sequences from MCTS benchmark results.
Creates professional Nature-format plots showing distribution of skip, only-skip, 
and other operations in tier4 optimal sequences across models and benchmarks.
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))


import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set publication-quality style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Color palette - seaborn viridis
_viridis = sns.color_palette("viridis", 2)
COLORS = {
    'has_skip': _viridis[0],
    'only_skip': _viridis[0],
    'other_ops': _viridis[1],
}

# Model configurations
MODEL_CONFIGS = {
    'qwen25_0.5b': {
        'name': 'Qwen2.5-0.5B',
        'benchmarks': {
            'winogrande': 'benchmark_mcts_winogrande_20260226-111323_snapshot.json',
            'boolq': 'benchmark_mcts_boolq_20260226-182754_snapshot.json',
            'bigbench': 'benchmark_mcts_bigbench_all_20260228-015212_snapshot.json',
            'commonsenseqa': 'benchmark_mcts_commonsenseqa_20260226-163614_snapshot.json',
            'mmlu': 'benchmark_mcts_mmlu_all_20260226-141452_snapshot.json',
        }
    },
    'qwen25_0.5b_fixed': {
        'name': 'Qwen2.5-0.5B (fixed)',
        'benchmarks': {
            'winogrande': 'fixed_benchmark_mcts_winogrande_qwen25_0.5b_20260305-142122_snapshot.json',
            'boolq': 'benchmark_mcts_boolq_20260306-100226_snapshot.json',
            'bigbench': 'benchmark_mcts_bigbench_boolean_expressions_20260306-090935_snapshot.json',
            'commonsenseqa': 'fixed_benchmark_mcts_commonsenseqa_qwen25_0.5b_20260305-095955_snapshot.json',
            'mmlu': 'benchmark_mcts_mmlu_all_20260306-105212_snapshot.json',
        },
        'search_dirs': [],  # try model_dir and predictions root
    },
    'qwen25_7b': {
        'name': 'Qwen2.5-7B',
        'benchmarks': {
            'winogrande': 'benchmark_mcts_winogrande_20260226-110816_snapshot.json',
            'boolq': 'benchmark_mcts_boolq_20260302-201239_snapshot.json',
            'bigbench': 'benchmark_mcts_bigbench_all_20260303-135714_snapshot.json',
            'commonsenseqa': 'benchmark_mcts_commonsenseqa_20260302-135451_snapshot.json',
            'mmlu': 'benchmark_mcts_mmlu_all_20260226-195110_snapshot.json',
        }
    },
}

# Benchmark display names
BENCHMARK_NAMES = {
    'winogrande': 'WinoGrande',
    'boolq': 'BoolQ',
    'bigbench': 'BigBench',
    'bigbench_boolean_expressions': 'BigBench (boolean)',
    'commonsenseqa': 'CommonsenseQA',
    'mmlu': 'MMLU',
}


def analyze_sequence(seq: List[int]) -> Dict[str, int]:
    """
    Analyze a sequence to count skip and other operations.
    
    Args:
        seq: List of layer indices
        
    Returns:
        Dictionary with counts of different operation types
    """
    skip_count = 0
    duplicate_count = 0
    
    for i, layer_id in enumerate(seq):
        if layer_id == -1:
            skip_count += 1
        elif i > 0 and layer_id == seq[i-1]:
            duplicate_count += 1
    
    has_skip = skip_count > 0
    only_skip = skip_count > 0 and duplicate_count == 0
    has_other = duplicate_count > 0
    
    return {
        'skip_count': skip_count,
        'duplicate_count': duplicate_count,
        'has_skip': has_skip,
        'only_skip': only_skip,
        'has_other': has_other,
    }


def load_tier4_sequences(filepath: Path, top_k: int = None) -> List[Dict]:
    """Load top-k tier4 sequences from a benchmark file. top_k=None means all."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    tier4 = data.get('tier4', [])
    
    # Sort by accuracy (descending)
    tier4_sorted = sorted(tier4, key=lambda x: x.get('accuracy', 0), reverse=True)
    
    return tier4_sorted[:top_k] if top_k is not None else tier4_sorted


def collect_data(predictions_dir: Path, top_k: int = 5, model_filter: str = None) -> Dict:
    """
    Collect operation statistics from all models and benchmarks.
    
    Returns:
        Dictionary with nested structure: model -> benchmark -> statistics
    """
    results = defaultdict(lambda: defaultdict(list))
    
    for model_key, model_config in MODEL_CONFIGS.items():
        if model_filter is not None and model_key != model_filter:
            continue
        model_dir = predictions_dir / model_key
        search_in_root = model_config.get('search_dirs') is not None  # e.g. qwen25_0.5b_fixed
        
        for bench_key, bench_file in model_config['benchmarks'].items():
            filepath = model_dir / bench_file
            if not filepath.exists() and search_in_root:
                filepath = predictions_dir / bench_file
            if not filepath.exists():
                print(f"Warning: File not found: {bench_file}")
                continue
            
            try:
                sequences = load_tier4_sequences(filepath, top_k=top_k)
                
                for seq_data in sequences:
                    seq = seq_data['seq']
                    analysis = analyze_sequence(seq)
                    
                    results[model_key][bench_key].append({
                        'sequence': seq,
                        'accuracy': seq_data.get('accuracy', 0),
                        'delta': seq_data.get('delta', 0),
                        **analysis
                    })
                    
                print(f"✓ Loaded {len(sequences)} sequences from {model_key}/{bench_key}")
                
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return results


def categorize_sequences(sequences: List[Dict]) -> Dict[str, int]:
    """
    Categorize sequences into: has_skip, only_skip, other_ops.
    
    Returns:
        Dictionary with counts for each category
    """
    counts = {
        'has_skip': 0,      # Has at least one skip (including those with other ops)
        'only_skip': 0,     # Has skips but no duplicate operations
        'other_ops': 0,     # Has duplicate operations (may also have skips)
    }
    
    for seq_data in sequences:
        if seq_data['has_other']:
            counts['other_ops'] += 1
        elif seq_data['has_skip']:
            counts['only_skip'] += 1
        # If neither has_skip nor has_other, it's a baseline sequence (not counted)
    
    return counts


def plot_per_model(results: Dict, output_dir: Path):
    """Create individual plots for each model."""
    
    for model_key, model_data in results.items():
        model_name = MODEL_CONFIGS[model_key]['name']
        
        # Prepare data
        benchmarks = []
        has_skip_counts = []
        only_skip_counts = []
        other_ops_counts = []
        
        for bench_key in ['winogrande', 'boolq', 'bigbench', 'commonsenseqa', 'mmlu']:
            if bench_key not in model_data:
                continue
                
            benchmarks.append(BENCHMARK_NAMES[bench_key])
            counts = categorize_sequences(model_data[bench_key])
            
            # Stack: bottom = only_skip, middle = has_skip (but not only), top = other_ops
            only_skip_counts.append(counts['only_skip'])
            # has_skip includes those with other ops, so we need to separate them
            has_skip_without_other = sum(1 for s in model_data[bench_key] 
                                         if s['has_skip'] and not s['has_other'])
            other_with_skip = sum(1 for s in model_data[bench_key] 
                                  if s['has_skip'] and s['has_other'])
            
            has_skip_counts.append(has_skip_without_other)
            other_ops_counts.append(counts['other_ops'])
        
        if not benchmarks:
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = np.arange(len(benchmarks))
        width = 0.6
        
        # Create stacked bar chart
        p1 = ax.bar(x, only_skip_counts, width, label='Only Skip', 
                    color=COLORS['only_skip'], edgecolor='black', linewidth=0.5)
        p2 = ax.bar(x, other_ops_counts, width, bottom=only_skip_counts,
                    label='Other Operations', color=COLORS['other_ops'], 
                    edgecolor='black', linewidth=0.5)
        
        # Styling
        total_max = max(o + ot for o, ot in zip(only_skip_counts, other_ops_counts))
        y_max = max(total_max * 1.15, 5)
        ax.set_ylabel('Number of Sequences (Tier-4)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Benchmark', fontsize=11, fontweight='bold')
        ax.set_title(f'Operation Distribution in Optimal Sequences\n{model_name}', 
                     fontsize=12, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=30, ha='right')
        ax.set_ylim(0, y_max)
        ax.legend(frameon=True, loc='upper right', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add value labels on bars
        for i, (o, ot) in enumerate(zip(only_skip_counts, other_ops_counts)):
            if o > 0:
                ax.text(i, o/2, str(o), ha='center', va='center', 
                       fontsize=8, fontweight='bold', color='white')
            if ot > 0:
                ax.text(i, o + ot/2, str(ot), ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')
        
        plt.tight_layout()
        
        output_file = output_dir / f'operations_distribution_{model_key}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_file}")


def _aggregate_title_models(results: Dict) -> str:
    """Build model-family string for aggregate plot title (e.g. 'Qwen models' or 'Qwen and Llama models')."""
    model_keys = list(results.keys())
    has_qwen = any('qwen' in k.lower() for k in model_keys)
    has_llama = any('llama' in k.lower() for k in model_keys)
    if has_qwen and has_llama:
        return 'Qwen and Llama models'
    if has_llama:
        return 'Llama models'
    return 'Qwen models'


def plot_aggregate(results: Dict, output_dir: Path):
    """Create aggregate plot across all models."""
    
    models_str = _aggregate_title_models(results)
    
    # Aggregate data across models
    benchmark_data = defaultdict(lambda: {'only_skip': 0, 'other_ops': 0})
    
    for model_key, model_data in results.items():
        for bench_key, sequences in model_data.items():
            counts = categorize_sequences(sequences)
            # Separate only_skip from those with other operations
            only_skip = sum(1 for s in sequences if s['has_skip'] and not s['has_other'])
            other_ops = counts['other_ops']
            
            benchmark_data[bench_key]['only_skip'] += only_skip
            benchmark_data[bench_key]['other_ops'] += other_ops
    
    # Prepare data for plotting
    benchmarks = []
    only_skip_counts = []
    other_ops_counts = []
    
    for bench_key in ['winogrande', 'boolq', 'bigbench', 'commonsenseqa', 'mmlu']:
        if bench_key not in benchmark_data:
            continue
        benchmarks.append(BENCHMARK_NAMES[bench_key])
        only_skip_counts.append(benchmark_data[bench_key]['only_skip'])
        other_ops_counts.append(benchmark_data[bench_key]['other_ops'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(benchmarks))
    width = 0.6
    
    # Create stacked bar chart
    p1 = ax.bar(x, only_skip_counts, width, label='Only Skip', 
                color=COLORS['only_skip'], edgecolor='black', linewidth=0.5)
    p2 = ax.bar(x, other_ops_counts, width, bottom=only_skip_counts,
                label='Other Operations', color=COLORS['other_ops'], 
                edgecolor='black', linewidth=0.5)
    
    # Styling
    ax.set_ylabel('Number of Sequences (Tier-4 × Models)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_title(f'Operation Distribution in Optimal Sequences\nAggregated Across {models_str}', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=30, ha='right')
    total_max = max(o + ot for o, ot in zip(only_skip_counts, other_ops_counts))
    ax.set_ylim(0, max(total_max * 1.15, 5))
    ax.legend(frameon=True, loc='upper right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value labels on bars
    for i, (o, ot) in enumerate(zip(only_skip_counts, other_ops_counts)):
        if o > 0:
            ax.text(i, o/2, str(o), ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='white')
        if ot > 0:
            ax.text(i, o + ot/2, str(ot), ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
    
    plt.tight_layout()
    
    output_file = output_dir / 'operations_distribution_aggregate.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_heatmap(results: Dict, output_dir: Path):
    """Create a heatmap showing operation patterns across models and benchmarks."""
    
    # Prepare data matrix (only models we have data for)
    models = list(results.keys())
    benchmarks = ['winogrande', 'boolq', 'bigbench', 'commonsenseqa', 'mmlu']
    
    # Three matrices: only_skip, other_ops, and percentage
    only_skip_matrix = np.zeros((len(models), len(benchmarks)))
    other_ops_matrix = np.zeros((len(models), len(benchmarks)))
    
    for i, model_key in enumerate(models):
        for j, bench_key in enumerate(benchmarks):
            if bench_key in results[model_key]:
                sequences = results[model_key][bench_key]
                only_skip = sum(1 for s in sequences if s['has_skip'] and not s['has_other'])
                other_ops = sum(1 for s in sequences if s['has_other'])
                
                only_skip_matrix[i, j] = only_skip
                other_ops_matrix[i, j] = other_ops
    
    vmax = max(int(np.max(only_skip_matrix)), int(np.max(other_ops_matrix)), 5)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Plot 1: Only Skip
    sns.heatmap(only_skip_matrix, annot=True, fmt='.0f', cmap='viridis', 
                xticklabels=[BENCHMARK_NAMES[b] for b in benchmarks],
                yticklabels=[MODEL_CONFIGS[m]['name'] for m in models],
                cbar_kws={'label': 'Count'}, ax=ax1, vmin=0, vmax=vmax,
                linewidths=0.5, linecolor='gray')
    ax1.set_title('Only Skip Operations', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Benchmark', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Model', fontsize=11, fontweight='bold')
    
    # Plot 2: Other Operations
    sns.heatmap(other_ops_matrix, annot=True, fmt='.0f', cmap='viridis', 
                xticklabels=[BENCHMARK_NAMES[b] for b in benchmarks],
                yticklabels=[MODEL_CONFIGS[m]['name'] for m in models],
                cbar_kws={'label': 'Count'}, ax=ax2, vmin=0, vmax=vmax,
                linewidths=0.5, linecolor='gray')
    ax2.set_title('Other Operations (Duplicates)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Benchmark', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Model', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / 'operations_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def create_summary_report(results: Dict, output_dir: Path):
    """Create a text summary of the analysis."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("OPTIMAL MODULE SEQUENCE OPERATION ANALYSIS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Analysis of all tier4 optimal sequences across models and benchmarks")
    report_lines.append("")
    
    for model_key, model_data in results.items():
        model_name = MODEL_CONFIGS[model_key]['name']
        report_lines.append(f"\n{model_name}")
        report_lines.append("-" * 80)
        
        for bench_key, sequences in sorted(model_data.items()):
            bench_name = BENCHMARK_NAMES[bench_key]
            counts = categorize_sequences(sequences)
            
            only_skip = sum(1 for s in sequences if s['has_skip'] and not s['has_other'])
            other_ops = counts['other_ops']
            
            report_lines.append(f"\n  {bench_name}:")
            report_lines.append(f"    Total sequences: {len(sequences)}")
            report_lines.append(f"    Only Skip:       {only_skip}")
            report_lines.append(f"    Other Operations: {other_ops}")
            
            # Show sample sequences
            for i, seq_data in enumerate(sequences[:3], 1):
                seq_str = str(seq_data['sequence'])
                if len(seq_str) > 60:
                    seq_str = seq_str[:60] + "...]"
                report_lines.append(f"    Seq {i}: {seq_str}")
                report_lines.append(f"           Acc: {seq_data['accuracy']:.3f}, "
                                  f"Skips: {seq_data['skip_count']}, "
                                  f"Dups: {seq_data['duplicate_count']}")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("AGGREGATE STATISTICS")
    report_lines.append("=" * 80)
    
    # Aggregate stats
    total_sequences = sum(len(seqs) for model_data in results.values() 
                         for seqs in model_data.values())
    total_only_skip = sum(1 for model_data in results.values() 
                         for sequences in model_data.values()
                         for s in sequences if s['has_skip'] and not s['has_other'])
    total_other_ops = sum(1 for model_data in results.values() 
                         for sequences in model_data.values()
                         for s in sequences if s['has_other'])
    
    report_lines.append(f"\nTotal sequences analyzed: {total_sequences}")
    report_lines.append(f"Only Skip: {total_only_skip} ({100*total_only_skip/total_sequences:.1f}%)")
    report_lines.append(f"Other Operations: {total_other_ops} ({100*total_other_ops/total_sequences:.1f}%)")
    
    report_text = "\n".join(report_lines)
    
    # Save report
    report_file = output_dir / 'operations_analysis_report.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"✓ Saved: {report_file}")
    print("\n" + report_text)


def main():
    """Main execution function."""
    import argparse
    parser = argparse.ArgumentParser(description='Analyze optimal operations in MCTS benchmark results')
    parser.add_argument('--model', type=str, default=None,
                       help='Only run for this model (e.g. qwen25_0.5b_fixed). Default: all models.')
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    predictions_dir = script_dir
    output_dir = script_dir / 'optimal_operations_analysis'
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("ANALYZING OPTIMAL MODULE SEQUENCES")
    print("=" * 80)
    print(f"\nPredictions directory: {predictions_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Collect data
    print("Collecting data from benchmark files...")
    results = collect_data(predictions_dir, top_k=None, model_filter=args.model)
    
    if not results:
        print("\n❌ No data collected. Please check file paths.")
        return
    
    print(f"\n✓ Collected data from {len(results)} models")
    print()
    
    # Generate plots
    print("Generating plots...")
    plot_per_model(results, output_dir)
    plot_aggregate(results, output_dir)
    plot_heatmap(results, output_dir)
    
    # Create summary report
    print("\nGenerating summary report...")
    create_summary_report(results, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - operations_distribution_qwen25_0.5b.png")
    print("  - operations_distribution_qwen25_7b.png")
    print("  - operations_distribution_aggregate.png")
    print("  - operations_heatmap.png")
    print("  - operations_analysis_report.txt")


if __name__ == '__main__':
    main()
