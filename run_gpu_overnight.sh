#!/bin/bash
# GPU overnight experiment suite.
# Each experiment runs for 30 min with the same settings.
# Run on a machine with CUDA GPU or Apple Silicon.
# Logs go to logs/exp_*.txt — watch with: tail -f logs/exp_*.txt
set -e

export VAL_MAX_TOKENS=1000000 ITERATIONS=100000 MAX_WALLCLOCK_SECONDS=1800
export TRAIN_BATCH_TOKENS=4096 GRAD_ACCUM_STEPS=1 TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=0
export VAL_BATCH_SIZE=4096 MLX_MAX_MICROBATCH_TOKENS=4096

echo "Starting $(date)"
echo "===================="

# 1. Baseline + shift (for comparison on this machine)
echo "[1/10] Baseline + shift"
RUN_ID=gpu python3 train_gpt_mlx.py

# 2. Multi-shift (distances 1-4)
echo "[2/10] Multi-shift"
RUN_ID=gpu python3 experiments_v2.py multi_shift

# 3. Conv1d kernel=4
echo "[3/10] Conv kernel=4"
RUN_ID=gpu python3 experiments_v2.py conv --conv-kernel 4

# 4. Funnel (wide edge, narrow mid)
echo "[4/10] Funnel (edge=4x, mid=1x)"
RUN_ID=gpu python3 experiments_v2.py funnel --edge-mult 4 --mid-mult 1

# 5. Funnel variant (edge=3x, mid=2x)
echo "[5/10] Funnel (edge=3x, mid=2x)"
RUN_ID=gpu2 python3 experiments_v2.py funnel --edge-mult 3 --mid-mult 2

# 6. MLP-only middle (no attention layers 2-6)
echo "[6/10] MLP-only middle"
RUN_ID=gpu python3 experiments_v2.py mlp_only_mid

# 7. Depth recurrence (4 unique × 2 loops = 8 effective)
echo "[7/10] Depth recur 4×2"
RUN_ID=gpu python3 experiments_v2.py depth_recur --num-unique 4 --num-loops 2

# 8. Depth recurrence (3 unique × 3 loops = 9 effective)
echo "[8/10] Depth recur 3×3"
RUN_ID=gpu python3 experiments_v2.py depth_recur --num-unique 3 --num-loops 3

# 9. Depth recurrence (5 unique × 2 loops = 10 effective)
echo "[9/10] Depth recur 5×2"
RUN_ID=gpu python3 experiments_v2.py depth_recur --num-unique 5 --num-loops 2

# 10. Attention residual (shared QK, per-layer V)
echo "[10/13] Attn residual"
RUN_ID=gpu python3 experiments.py attn_resid

# 11. Stacked: shared QK + shift + conv4 + funnel(3/2)
echo "[11/13] Stacked (3/2 funnel, conv4)"
RUN_ID=gpu python3 experiments_v2.py stacked --edge-mult 3 --mid-mult 2 --conv-kernel 4

# 12. Stacked: shared QK + shift + conv4 + funnel(4/1)
echo "[12/13] Stacked (4/1 funnel, conv4)"
RUN_ID=gpu2 python3 experiments_v2.py stacked --edge-mult 4 --mid-mult 1 --conv-kernel 4

# 13. Stacked: shared QK + shift + conv2 + even MLP
echo "[13/13] Stacked (2/2, conv2)"
RUN_ID=gpu3 python3 experiments_v2.py stacked --edge-mult 2 --mid-mult 2 --conv-kernel 2

echo "===================="
echo "All done $(date)"
echo "Results:"
echo "--------"
grep "final_int8_zlib_roundtrip_exact" logs/exp_*.txt logs/attn_resid_*.txt logs/baseline_shift_*.txt 2>/dev/null | sort -t: -k3 -n
