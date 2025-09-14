"""
Evaluation metrics and visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False,
                         title='Confusion Matrix', figsize=(8, 6)):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the matrix
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib.figure.Figure
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', ax=ax)

    if class_names is not None:
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_training_history(history, metrics=['loss', 'accuracy'], figsize=(12, 4)):
    """
    Plot training history.

    Args:
        history: Keras training history
        metrics: List of metrics to plot
        figsize: Figure size

    Returns:
        matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Plot training metric
        if metric in history.history:
            ax.plot(history.history[metric], label=f'Training {metric}')

        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            ax.plot(history.history[val_metric], label=f'Validation {metric}')

        ax.set_title(f'{metric.capitalize()} vs Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def calculate_confidence_metrics(y_true, y_pred_proba, confidence_threshold=0.9):
    """
    Calculate confidence-based metrics.

    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        confidence_threshold: Threshold for high-confidence predictions

    Returns:
        Dictionary with confidence metrics
    """
    max_probs = np.max(y_pred_proba, axis=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # High-confidence predictions
    high_conf_mask = max_probs >= confidence_threshold
    high_conf_accuracy = np.mean(y_true[high_conf_mask] == y_pred[high_conf_mask]) if np.sum(high_conf_mask) > 0 else 0
    high_conf_ratio = np.mean(high_conf_mask)

    # Calculate BvSB (Best vs Second Best) scores
    sorted_probs = np.sort(y_pred_proba, axis=1)[:, ::-1]
    bvsb_scores = sorted_probs[:, 0] - sorted_probs[:, 1]

    metrics = {
        'mean_confidence': np.mean(max_probs),
        'high_confidence_accuracy': high_conf_accuracy,
        'high_confidence_ratio': high_conf_ratio,
        'mean_bvsb': np.mean(bvsb_scores),
        'confidence_scores': max_probs,
        'bvsb_scores': bvsb_scores
    }

    return metrics


def evaluate_class_performance(y_true, y_pred, class_names=None):
    """
    Evaluate per-class performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Dictionary with per-class metrics
    """
    # Generate classification report
    if class_names is not None:
        target_names = class_names
    else:
        target_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]

    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True
    )

    # Calculate per-class accuracy
    unique_classes = np.unique(y_true)
    per_class_accuracy = {}

    for cls in unique_classes:
        mask = y_true == cls
        if np.sum(mask) > 0:
            accuracy = np.mean(y_pred[mask] == cls)
            class_name = target_names[cls] if cls < len(target_names) else f'Class {cls}'
            per_class_accuracy[class_name] = accuracy

    return {
        'classification_report': report,
        'per_class_accuracy': per_class_accuracy
    }


def plot_confidence_distribution(confidence_scores, bins=50, title='Confidence Score Distribution'):
    """
    Plot distribution of confidence scores.

    Args:
        confidence_scores: Array of confidence scores
        bins: Number of histogram bins
        title: Plot title

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(confidence_scores, bins=bins, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(confidence_scores), color='red', linestyle='--',
               label=f'Mean: {np.mean(confidence_scores):.3f}')

    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig