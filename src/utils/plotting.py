"""
Plotting and visualization utilities for experimental results.
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_results(results, output_dir='./', save_plots=True):
    """
    Plot comprehensive training results from experiments.

    Args:
        results: Dictionary containing training results
        output_dir: Directory to save plots
        save_plots: Whether to save plots to files

    Returns:
        Dictionary of matplotlib figures
    """
    figures = {}

    # Accuracy comparison
    if all(key in results for key in ['acc_n', 'acc_m']):
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(results['acc_n'], label="Original N classes", linewidth=2)
        if 'acc_n_stratified' in results and results['acc_n_stratified']:
            ax.plot(results['acc_n_stratified'], label="Stratified Sampling", linewidth=2)
        if 'acc_n_subsampling' in results and results['acc_n_subsampling']:
            ax.plot(results['acc_n_subsampling'], label="Subsampling", linewidth=2)
        ax.plot(results['acc_m'], label="M classes", linewidth=2)

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Accuracy Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        figures['accuracy_comparison'] = fig

        if save_plots:
            plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')

    # Loss comparison
    if all(key in results for key in ['loss_n', 'loss_m']):
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(results['loss_n'], label="Original N classes", linewidth=2)
        if 'loss_n_stratified' in results and results['loss_n_stratified']:
            ax.plot(results['loss_n_stratified'], label="Stratified Sampling", linewidth=2)
        if 'loss_n_subsampling' in results and results['loss_n_subsampling']:
            ax.plot(results['loss_n_subsampling'], label="Subsampling", linewidth=2)
        ax.plot(results['loss_m'], label="M classes", linewidth=2)

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        figures['loss_comparison'] = fig

        if save_plots:
            plt.savefig(os.path.join(output_dir, 'loss_comparison.png'), dpi=300, bbox_inches='tight')

    # Training time analysis
    if 'training_times' in results and results['training_times']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Time per step
        training_times = results['training_times']
        ax1.plot(training_times, label='Training Time per Step', linewidth=2)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Training Time per Step')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Time statistics
        stats = {
            'Mean': np.mean(training_times),
            'Median': np.median(training_times),
            'Max': np.max(training_times),
            'Min': np.min(training_times),
            '90th Percentile': np.percentile(training_times, 90)
        }

        ax2.bar(stats.keys(), stats.values())
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Training Time Statistics')
        ax2.tick_params(axis='x', rotation=45)

        # Add statistics text
        stats_text = "\n".join([f"{k}: {v:.3f}s" for k, v in stats.items()])
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        figures['training_time'] = fig

        if save_plots:
            plt.savefig(os.path.join(output_dir, 'training_time.png'), dpi=300, bbox_inches='tight')

    return figures


def plot_data_from_files(file_paths, labels=None, title="Data Comparison",
                        xlabel="Training Steps", ylabel="Value", output_path=None):
    """
    Plot data loaded from text files.

    Args:
        file_paths: List of file paths or dictionary mapping labels to file paths
        labels: List of labels for each file (if file_paths is a list)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        output_path: Path to save plot

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    if isinstance(file_paths, dict):
        # Dictionary format
        for label, file_path in file_paths.items():
            if os.path.exists(file_path):
                data = []
                with open(file_path, 'r') as f:
                    for line in f:
                        data.append(float(line.strip()))
                ax.plot(data, label=label, linewidth=2)
    else:
        # List format
        if labels is None:
            labels = [f"Data {i+1}" for i in range(len(file_paths))]

        for file_path, label in zip(file_paths, labels):
            if os.path.exists(file_path):
                data = []
                with open(file_path, 'r') as f:
                    for line in f:
                        data.append(float(line.strip()))
                ax.plot(data, label=label, linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def create_experiment_summary_plots(results_dict, output_dir='./plots/'):
    """
    Create a comprehensive summary of experimental results.

    Args:
        results_dict: Dictionary containing multiple experiment results
        output_dir: Directory to save plots

    Returns:
        Dictionary of created figures
    """
    os.makedirs(output_dir, exist_ok=True)
    figures = {}

    # Create individual plots for each experiment
    for exp_name, results in results_dict.items():
        print(f"Creating plots for {exp_name}")
        exp_figures = plot_training_results(
            results,
            output_dir=os.path.join(output_dir, exp_name),
            save_plots=True
        )
        figures[exp_name] = exp_figures

    return figures


def plot_confidence_histogram(confidence_scores, bins=30, title="Confidence Score Distribution"):
    """
    Plot histogram of model confidence scores.

    Args:
        confidence_scores: Array of confidence scores
        bins: Number of histogram bins
        title: Plot title

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(confidence_scores, bins=bins, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(confidence_scores), color='red', linestyle='--',
               label=f'Mean: {np.mean(confidence_scores):.3f}')
    ax.axvline(np.median(confidence_scores), color='orange', linestyle='--',
               label=f'Median: {np.median(confidence_scores):.3f}')

    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_class_distribution(y_data, class_names=None, title="Class Distribution"):
    """
    Plot distribution of classes in the dataset.

    Args:
        y_data: Array of class labels
        class_names: List of class names
        title: Plot title

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    class_counts = np.bincount(y_data.astype(int))
    class_indices = np.arange(len(class_counts))

    if class_names is None:
        class_names = [f'Class {i}' for i in class_indices]

    bars = ax.bar(class_indices, class_counts, alpha=0.7)
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    ax.set_title(title)
    ax.set_xticks(class_indices)
    ax.set_xticklabels(class_names, rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    return fig