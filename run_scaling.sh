#!/bin/bash
# LoRA routing scaling exploration.
# 9 experiments × 30 min = ~4.5 hours.
set -e

export VAL_MAX_TOKENS=1000000 ITERATIONS=100000 MAX_WALLCLOCK_SECONDS=1800
export TRAIN_BATCH_TOKENS=4096 GRAD_ACCUM_STEPS=1 TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=0
export VAL_BATCH_SIZE=4096 MLX_MAX_MICROBATCH_TOKENS=4096

# --- AXIS 1: RANK (width per LoRA layer) ---
# stem=3, recurrence=6, pool=8×8, vary rank

echo "=== [1/9] RANK 32 ==="
RUN_ID=scale_rank32 python3 experiments.py lora_routed --rank 32

echo "=== [2/9] RANK 128 ==="
RUN_ID=scale_rank128 python3 experiments.py lora_routed --rank 128

echo "=== [3/9] RANK 192 ==="
RUN_ID=scale_rank192 python3 experiments.py lora_routed --rank 192

# --- AXIS 2: RECURRENCE (depth) ---
# stem=3, rank=64, pool=8×8, vary recurrence

echo "=== [4/9] RECURRENCE 3 ==="
RUN_ID=scale_rec3 python3 experiments.py lora_routed --num-recurrence 3

echo "=== [5/9] RECURRENCE 12 ==="
RUN_ID=scale_rec12 python3 experiments.py lora_routed --num-recurrence 12

echo "=== [6/9] RECURRENCE 24 ==="
RUN_ID=scale_rec24 python3 experiments.py lora_routed --num-recurrence 24

# --- AXIS 3: POOL SIZE (A×B combos) ---
# stem=3, rank=64, recurrence=6, vary pool

echo "=== [7/9] POOL 4×4 ==="
RUN_ID=scale_pool4x4 python3 experiments.py lora_routed --num-a 4 --num-b 4

echo "=== [8/9] POOL 16×16 ==="
RUN_ID=scale_pool16x16 python3 experiments.py lora_routed --num-a 16 --num-b 16

# --- AXIS 4: STEM ---
# rank=64, recurrence=6, pool=8×8, vary stem

echo "=== [9/9] STEM 1 ==="
RUN_ID=scale_stem1 python3 experiments.py lora_routed --num-stem 1

echo "========================================="
echo "ALL SCALING EXPERIMENTS COMPLETE"
echo "========================================="
