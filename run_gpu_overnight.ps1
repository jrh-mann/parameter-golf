# GPU overnight experiment suite (PowerShell version).
# Each experiment runs for 30 min with the same settings.
# Logs go to logs/exp_*.txt — watch with: Get-Content -Wait logs/exp_*.txt
$ErrorActionPreference = "Stop"

$env:VAL_MAX_TOKENS = "1000000"
$env:ITERATIONS = "100000"
$env:MAX_WALLCLOCK_SECONDS = "1800"
$env:TRAIN_BATCH_TOKENS = "4096"
$env:GRAD_ACCUM_STEPS = "1"
$env:TRAIN_LOG_EVERY = "50"
$env:VAL_LOSS_EVERY = "0"
$env:VAL_BATCH_SIZE = "4096"
$env:MLX_MAX_MICROBATCH_TOKENS = "4096"

Write-Host "Starting $(Get-Date)"
Write-Host "===================="

# 1. Baseline + shift (for comparison on this machine)
Write-Host "[1/13] Baseline + shift"
$env:RUN_ID = "gpu"
python train_gpt_mlx.py
if ($LASTEXITCODE -ne 0) { throw "Experiment 1 failed" }

# 2. Multi-shift (distances 1-4)
Write-Host "[2/13] Multi-shift"
$env:RUN_ID = "gpu"
python experiments_v2.py multi_shift
if ($LASTEXITCODE -ne 0) { throw "Experiment 2 failed" }

# 3. Conv1d kernel=4
Write-Host "[3/13] Conv kernel=4"
$env:RUN_ID = "gpu"
python experiments_v2.py conv --conv-kernel 4
if ($LASTEXITCODE -ne 0) { throw "Experiment 3 failed" }

# 4. Funnel (wide edge, narrow mid)
Write-Host "[4/13] Funnel (edge=4x, mid=1x)"
$env:RUN_ID = "gpu"
python experiments_v2.py funnel --edge-mult 4 --mid-mult 1
if ($LASTEXITCODE -ne 0) { throw "Experiment 4 failed" }

# 5. Funnel variant (edge=3x, mid=2x)
Write-Host "[5/13] Funnel (edge=3x, mid=2x)"
$env:RUN_ID = "gpu2"
python experiments_v2.py funnel --edge-mult 3 --mid-mult 2
if ($LASTEXITCODE -ne 0) { throw "Experiment 5 failed" }

# 6. MLP-only middle (no attention layers 2-6)
Write-Host "[6/13] MLP-only middle"
$env:RUN_ID = "gpu"
python experiments_v2.py mlp_only_mid
if ($LASTEXITCODE -ne 0) { throw "Experiment 6 failed" }

# 7. Depth recurrence (4 unique x 2 loops = 8 effective)
Write-Host "[7/13] Depth recur 4x2"
$env:RUN_ID = "gpu"
python experiments_v2.py depth_recur --num-unique 4 --num-loops 2
if ($LASTEXITCODE -ne 0) { throw "Experiment 7 failed" }

# 8. Depth recurrence (3 unique x 3 loops = 9 effective)
Write-Host "[8/13] Depth recur 3x3"
$env:RUN_ID = "gpu"
python experiments_v2.py depth_recur --num-unique 3 --num-loops 3
if ($LASTEXITCODE -ne 0) { throw "Experiment 8 failed" }

# 9. Depth recurrence (5 unique x 2 loops = 10 effective)
Write-Host "[9/13] Depth recur 5x2"
$env:RUN_ID = "gpu"
python experiments_v2.py depth_recur --num-unique 5 --num-loops 2
if ($LASTEXITCODE -ne 0) { throw "Experiment 9 failed" }

# 10. Attention residual (shared QK, per-layer V)
Write-Host "[10/13] Attn residual"
$env:RUN_ID = "gpu"
python experiments.py attn_resid
if ($LASTEXITCODE -ne 0) { throw "Experiment 10 failed" }

# 11. Stacked: shared QK + shift + conv4 + funnel(3/2)
Write-Host "[11/13] Stacked (3/2 funnel, conv4)"
$env:RUN_ID = "gpu"
python experiments_v2.py stacked --edge-mult 3 --mid-mult 2 --conv-kernel 4
if ($LASTEXITCODE -ne 0) { throw "Experiment 11 failed" }

# 12. Stacked: shared QK + shift + conv4 + funnel(4/1)
Write-Host "[12/13] Stacked (4/1 funnel, conv4)"
$env:RUN_ID = "gpu2"
python experiments_v2.py stacked --edge-mult 4 --mid-mult 1 --conv-kernel 4
if ($LASTEXITCODE -ne 0) { throw "Experiment 12 failed" }

# 13. Stacked: shared QK + shift + conv2 + even MLP
Write-Host "[13/13] Stacked (2/2, conv2)"
$env:RUN_ID = "gpu3"
python experiments_v2.py stacked --edge-mult 2 --mid-mult 2 --conv-kernel 2
if ($LASTEXITCODE -ne 0) { throw "Experiment 13 failed" }

Write-Host "===================="
Write-Host "All done $(Get-Date)"
Write-Host "Results:"
Write-Host "--------"
Get-ChildItem logs/exp_*.txt, logs/attn_resid_*.txt, logs/baseline_shift_*.txt -ErrorAction SilentlyContinue |
    Select-String "final_int8_zlib_roundtrip_exact" |
    Sort-Object { [double]($_ -split ':')[-1] }
