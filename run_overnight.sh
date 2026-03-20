#!/bin/bash
# Overnight experiment runner.
# Each experiment runs for 30 min (1800s) with the same settings.
set -e

COMMON="VAL_MAX_TOKENS=1000000 ITERATIONS=100000 MAX_WALLCLOCK_SECONDS=1800 \
TRAIN_BATCH_TOKENS=4096 GRAD_ACCUM_STEPS=1 TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=4096 MLX_MAX_MICROBATCH_TOKENS=4096"

echo "========================================="
echo "EXPERIMENT 1: LoRA routed (3 stem + 6 recurrence, rank64, 8A×8B)"
echo "========================================="
eval "$COMMON RUN_ID=overnight python3 experiments.py lora_routed"

echo "========================================="
echo "EXPERIMENT 2: Routed (3 stem + 4 experts × 6 recurrence)"
echo "========================================="
eval "$COMMON RUN_ID=overnight python3 experiments.py routed"

echo "========================================="
echo "EXPERIMENT 3: Proj token (single-pass)"
echo "========================================="
eval "$COMMON RUN_ID=overnight python3 experiments.py proj_token"

echo "========================================="
echo "EXPERIMENT 4: Baseline @ seq_len=512"
echo "========================================="
eval "$COMMON TRAIN_SEQ_LEN=512 RUN_ID=overnight_baseline_512 python3 train_gpt_mlx.py"

echo "========================================="
echo "EXPERIMENT 5: Mid-layer split@7"
echo "========================================="
eval "$COMMON NEURALESE_ENABLED=1 NEURALESE_SPLIT_LAYER=7 RUN_ID=overnight_split7 python3 train_gpt_mlx.py"

echo "========================================="
echo "EXPERIMENT 6: Mid-layer split@4"
echo "========================================="
eval "$COMMON NEURALESE_ENABLED=1 NEURALESE_SPLIT_LAYER=4 RUN_ID=overnight_split4 python3 train_gpt_mlx.py"

echo "========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================="
echo "Results in logs/overnight_*.txt and logs/*_overnight.txt"
