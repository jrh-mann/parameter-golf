#!/usr/bin/env python3
"""Plot train loss as relative difference to baseline."""
import os, re, glob
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

def interpolate_baseline(base_steps, base_losses, query_steps):
    """Interpolate baseline loss at query step positions."""
    return np.interp(query_steps, base_steps, base_losses)

baseline_path = "logs/baseline_30min.txt"
base_steps, base_losses = parse_log(baseline_path)

logs = sorted(f for f in
    glob.glob("logs/baseline_shift_*.txt") +
    glob.glob("logs/attn_resid_*.txt") +
    glob.glob("logs/exp_*.txt") +
    glob.glob("logs/ortho_*.txt")
    if "smoke" not in f and "test" not in f)

fig, ax = plt.subplots(figsize=(10, 6))
newest = max(logs, key=os.path.getmtime) if logs else None

for path in logs:
    steps, losses = parse_log(path)
    if len(steps) < 10:
        continue
    # Only plot where both have data
    mask = (steps >= base_steps[0]) & (steps <= base_steps[-1])
    if mask.sum() < 5:
        continue
    s, l = steps[mask], losses[mask]
    base_interp = interpolate_baseline(base_steps, base_losses, s)
    rel_diff = (l - base_interp) / base_interp * 100
    # Smooth to reduce noise
    w = max(1, len(rel_diff) // 30)
    if w > 1 and len(rel_diff) > w:
        kernel = np.ones(w) / w
        rel_diff = np.convolve(rel_diff, kernel, mode='valid')
        s = s[:len(rel_diff)]

    label = os.path.basename(path).replace(".txt", "").replace("baseline_", "").replace("exp_", "")
    is_highlight = newest and newest in path
    if is_highlight:
        ax.plot(s, rel_diff, label=label, linewidth=2.5, zorder=10)
    else:
        ax.plot(s, rel_diff, label=label, linewidth=1, alpha=0.5)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
ax.set_xscale("log")
ax.set_xlim(left=200)
ax.set_xlabel("Step (log scale)")
ax.set_ylabel("% difference from baseline (lower = better)")
ax.set_title("Train Loss Relative to Baseline")
ax.legend(fontsize=7, loc='lower left')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/relative_curve.png", dpi=150)
print("Saved to figures/relative_curve.png")
