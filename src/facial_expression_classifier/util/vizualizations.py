"""Plotting functions with seaborn."""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import structlog
from tensorflow.math import confusion_matrix

log = structlog.get_logger()

sns.set_theme(
    context="notebook",
    style="whitegrid",
    palette="deep",
    font_scale=1.0,
)


def plot_loss(
    df: pd.DataFrame,
    path: Path,
):
    """Plot training and validation loss."""
    df = df[["training_loss", "validation_loss"]]
    g = sns.relplot(df, kind="line", legend="auto")
    g.fig.suptitle(t="Training and validation loss", x=0.4, size=14, weight="bold")
    g.fig.subplots_adjust(top=0.9)
    g.set_xlabels("Epochs", size=12, weight="bold")
    g.set_ylabels("Loss", size=12, weight="bold")
    sns.move_legend(obj=g, loc="right", bbox_to_anchor=(0.7, 0.6), ncol=1, title=None, frameon=True)
    g.savefig(path)
    g.fig.clf()
    log.info(f"saved: {path.name}")


def plot_accuracy(
    df: pd.DataFrame,
    path: Path,
):
    """Plot training and validation loss."""
    df = df[["training_accuracy", "validation_accuracy"]]
    g = sns.relplot(df, kind="line", legend="auto")
    g.fig.suptitle(t="Training and validation accuracy", x=0.4, size=14, weight="bold")
    g.fig.subplots_adjust(top=0.9)
    g.set_xlabels("Epochs", size=12, weight="bold")
    g.set_ylabels("Accuracy", size=12, weight="bold")
    sns.move_legend(obj=g, loc="right", bbox_to_anchor=(0.65, 0.5), ncol=1, title=None, frameon=True)
    g.savefig(path)
    g.fig.clf()
    log.info(f"saved: {path.name}")


def plot_confusion_matrix(
    validation_labels: np.ndarray,
    validation_predictions: np.ndarray,
    class_names: List,
    path: Path,
):
    """Create confusion/error matrix."""
    predictions = np.argmax(validation_predictions, axis=1)
    confusion_mtx = confusion_matrix(
        labels=validation_labels,
        predictions=predictions,
    )
    ax = sns.heatmap(
        data=confusion_mtx,
        annot=True,
        cmap="mako",
        fmt="d",
        square=True,
    )
    ax.set_xticklabels(class_names, rotation=0, ha="center", size=11)
    ax.set_yticklabels(class_names, rotation=90, ha="center", size=11)
    ax.set_title("Confusion Matrix", size=12, weight="bold")
    ax.set_xlabel("Predicted Labels", weight="bold")
    ax.set_ylabel("True Labels", weight="bold")
    fig = ax.get_figure()
    fig.savefig(path)
    fig.clf()
    log.info(f"saved: {path.name}")
