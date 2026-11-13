# -*- coding: utf-8 -*-
"""Regenerates Figures 2 to 8 of the manuscript from the values reported in its tables.

Usage: python make_figures.py --out ../manuscript
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "savefig.dpi": 400, "savefig.bbox": "tight",
})
MODELS = ["BERT", "RoBERTa", "DistilBERT", "TinyBERT", "MiniLM"]


def ci_plot(path, title, means, lows, highs, xlim):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    y = np.arange(len(MODELS))[::-1]
    for yi, m, lo, hi in zip(y, means, lows, highs):
        ax.plot([lo, hi], [yi, yi], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot([m], [yi], "o", ms=8, color=BLUE, mec="white", mew=2)
        ax.text(1.02, yi, f"{m:.4f}  [{lo:.4f}, {hi:.4f}]", va="center", fontsize=8.5, color=INK,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(y); ax.set_yticklabels(MODELS, color=INK)
    ax.set_xlim(*xlim); ax.set_xlabel("Mean Macro F1-score with 95% confidence interval")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, loc="left")
    fig.tight_layout(rect=(0, 0, 0.80, 1)); fig.savefig(path); plt.close(fig)


def grouped(path, title, ylabel, eager, flash, fmt="{:.0f}"):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(MODELS)); w = 0.36
    ax.bar(x - w / 2 - 0.01, eager, w, color=BLUE, label="Eager attention", edgecolor="white", linewidth=1)
    ax.bar(x + w / 2 + 0.01, flash, w, color=ORANGE, label="FlashAttention-2 (verified)", edgecolor="white", linewidth=1)
    for xi, e, f in zip(x, eager, flash):
        ax.text(xi + w / 2 + 0.01, f, f"{100 * (f - e) / e:+.1f}%", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(MODELS, color=INK); ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(eager) * 1.15)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title(title, loc="left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def confusion(path, title, cm, labels, figsize):
    cm = np.asarray(cm)
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fe", "#86b6ef", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap=cmap)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]; dark = v > cm.max() * 0.45
            ax.text(j, i, str(v), ha="center", va="center", fontsize=10, color="white" if dark else INK)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30 if len(labels) > 2 else 0, ha="right" if len(labels) > 2 else "center", color=INK)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="figures"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # Table 7 of the manuscript (binary, released-benchmark protocol)
    ci_plot(os.path.join(a.out, "fig2_binary_ci.png"), "Binary task: Macro F1 over three seeds",
            [0.9941, 0.9889, 0.9941, 0.9774, 0.9911], [0.9892, 0.9840, 0.9923, 0.9699, 0.9893],
            [0.9990, 0.9938, 0.9959, 0.9849, 0.9930], (0.965, 1.0))
    # Table 9 (sarcasm-type task)
    ci_plot(os.path.join(a.out, "fig4_type_ci.png"), "Sarcasm-type task: Macro F1 over three seeds",
            [0.8301, 0.8078, 0.8273, 0.7058, 0.8092], [0.8122, 0.7852, 0.7790, 0.5528, 0.7540],
            [0.8480, 0.8304, 0.8755, 0.8588, 0.8645], (0.5, 1.0))
    confusion(os.path.join(a.out, "fig3_cm_roberta_binary.png"), "RoBERTa, binary task, seed 42 (n = 1,500)",
              [[976, 5], [13, 506]], ["Non-Sarcastic", "Sarcastic"], (4.6, 3.8))
    confusion(os.path.join(a.out, "fig5_cm_bert_type.png"), "BERT, sarcasm-type task, seed 42 (n = 519)",
              [[228, 8, 0, 17, 4], [7, 62, 3, 8, 0], [1, 1, 71, 0, 4], [0, 2, 0, 62, 1], [4, 2, 2, 3, 29]],
              ["Humor", "Irony", "Flattery", "Self-Deprecation", "Passive-Aggression"], (6.0, 5.0))
    # Table 11 (FlashAttention-2, sarcasm-type task, RTX 3060 Ti)
    grouped(os.path.join(a.out, "fig6_fa_training_time.png"), "Training time, sarcasm-type task (RTX 3060 Ti)",
            "Training time (s)", [122.14, 193.45, 66.06, 16.62, 70.99], [105.24, 151.12, 58.61, 14.09, 61.67])
    grouped(os.path.join(a.out, "fig7_fa_training_memory.png"), "Peak training memory, sarcasm-type task (RTX 3060 Ti)",
            "Peak allocated memory (MB)", [2567.2, 2436.2, 1439.0, 624.7, 1123.0], [2257.3, 2185.0, 1292.6, 511.8, 954.9])
    grouped(os.path.join(a.out, "fig8_fa_latency.png"), "Inference latency, sarcasm-type task (RTX 3060 Ti)",
            "Mean latency (ms)", [25.98, 17.21, 15.45, 7.18, 16.64], [21.26, 12.53, 13.19, 5.80, 13.76])


if __name__ == "__main__":
    main()
