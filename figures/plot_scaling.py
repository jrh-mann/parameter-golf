#!/usr/bin/env python3
"""Plot training loss scaling curves from log files."""
import os
import re
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_log(path):
    steps, losses = [], []
    with open(path) as f:
        for line in f:
            m = re.match(r"step:(\d+)/\d+ train_loss:([\d.]+)", line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return np.array(steps), np.array(losses)

def plot_runs(log_files, labels=None, output="figures/scaling_curve.png", highlight=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, path in enumerate(log_files):
        steps, losses = parse_log(path)
        if len(steps) == 0:
            continue
        label = labels[i] if labels else path.split("/")[-1].replace(".txt", "")
        is_highlight = highlight and highlight in path
        is_baseline = "baseline_30min" in path and "shift" not in path
        if is_highlight:
            ax.plot(steps, losses, label=label, linewidth=2.5, zorder=10)
        elif is_baseline:
            ax.plot(steps, losses, label=label, linewidth=2, alpha=0.8, zorder=5)
        else:
            ax.plot(steps, losses, label=label, linewidth=1, alpha=0.35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1000)
    ax.set_ylim(2, 5)
    ax.set_xlabel("Step (log scale)")
    ax.set_ylabel("Train Loss (log scale)")
    ax.set_title("Training Loss Scaling Curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved to {output}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        import glob as _glob
        logs = sorted(f for f in _glob.glob("logs/baseline_shift_*.txt") + _glob.glob("logs/attn_resid_*.txt") + _glob.glob("logs/exp_*.txt") + _glob.glob("logs/ortho_*.txt") if "smoke" not in f and "test" not in f)
        files = ["logs/baseline_30min.txt"] + logs
        labels = ["Baseline"] + [
            os.path.basename(f).replace("lora_routed_", "").replace(".txt", "")
            .replace("_overnight", " (base)").replace("_scale_", " ")
            for f in logs
        ]
        # Highlight the most recently modified log
        newest = max(logs, key=os.path.getmtime) if logs else None
        plot_runs(files, labels, highlight=os.path.basename(newest).replace(".txt","") if newest else None)
    else:
        plot_runs(sys.argv[1:])
