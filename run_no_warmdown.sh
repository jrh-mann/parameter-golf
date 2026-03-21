#!/bin/bash
set -e
export VAL_MAX_TOKENS=1000000 ITERATIONS=100000 MAX_WALLCLOCK_SECONDS=1800
export TRAIN_BATCH_TOKENS=4096 GRAD_ACCUM_STEPS=1 TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=0
export VAL_BATCH_SIZE=4096 MLX_MAX_MICROBATCH_TOKENS=4096 WARMDOWN_ITERS=0

echo "=== [1/3] Baseline (no warmdown) ==="
RUN_ID=no_wd python3 train_gpt_mlx.py

echo "=== [2/3] Attn Resid Opt (no warmdown) ==="
RUN_ID=no_wd python3 experiments.py attn_resid_opt

echo "=== [3/3] Forked Head v2 (no warmdown) ==="
RUN_ID=no_wd python3 experiments.py forked_head

echo "=== DONE ==="
grep "final_int8_zlib_roundtrip_exact" logs/baseline_no_wd.txt logs/exp_attn_resid_opt_fm3_no_wd.txt logs/exp_forked_head_v2_no_wd.txt 2>/dev/null
