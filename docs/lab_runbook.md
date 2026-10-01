# Lab Runbook - NetShield-FL

Step-by-step checklist for the lab session on the college lab PC
(i9-13950HX, RTX 4090, Windows, no Docker).

## What to carry

- [ ] USB stick (16+ GB) with `transfer/to_lab/` folder containing:
  - `repo.bundle` - git bundle of the full repo
  - `data_bundle.zip` - processed data + feature stats
  - `lab_setup.ps1` - standalone setup script
  - `MANIFEST.json` - checksums
  - `wheelhouse/` (optional, ~3 GB) - offline Python wheels

## Pre-flight (laptop, before leaving)

```powershell
# Ensure everything is committed
git status  # must be clean

# Pack for lab
powershell -ExecutionPolicy Bypass -File scripts\pack_for_lab.ps1

# Optional: build wheelhouse (if lab has no internet)
powershell -ExecutionPolicy Bypass -File scripts\build_wheelhouse.ps1

# Copy transfer/to_lab/ to USB stick
```

## Lab setup (~15 min)

```powershell
# 1. Plug in USB, open PowerShell

# 2. Run lab_setup.ps1 (works even without the repo on the machine)
powershell -ExecutionPolicy Bypass -File D:\transfer\to_lab\lab_setup.ps1 `
    -TransferDir D:\transfer\to_lab `
    -RepoRoot C:\dev\netshield-fl

# This will:
# - Verify checksums
# - Clone repo from bundle
# - Install Python env (online or from wheelhouse)
# - Unpack data
# - Verify CUDA GPU
# - Set NETSHIELD_PROFILE=lab
```

## Training (~4-8 hours estimated)

```powershell
cd C:\dev\netshield-fl

# Activate venv
.\.venv\Scripts\Activate.ps1
$env:NETSHIELD_PROFILE = "lab"

# Run everything (transcript saved to logs/)
powershell -ExecutionPolicy Bypass -File scripts\lab_train_all.ps1
```

### Expected step durations (RTX 4090, rough estimates)

| Step | Duration |
|------|----------|
| XGBoost full | 5-15 min |
| MLP full | 30-60 min |
| MLP no-IAT full | 30-60 min |
| FL grid (30 runs, 50 rounds each) | 3-6 hours |
| Export + pack | 5 min |

### Skipping steps

```powershell
# Skip already-completed or unnecessary steps
powershell -ExecutionPolicy Bypass -File scripts\lab_train_all.ps1 -Skip xgb,mlp_no_iat
```

### If the session ends early

1. The script is **resumable** - you can kill it and rerun. Each training
   step checks for existing results and checkpoints.
2. FL training checkpoints every 5 rounds. Rerunning resumes from the
   last checkpoint.
3. Even partial results are useful - `pack_results.ps1` is called at
   the end, but you can also run it manually at any point:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\pack_results.ps1
```

### If something goes wrong

- **Python not found**: Install per-user from python.org (no admin needed)
- **CUDA error**: Check `nvidia-smi` for GPU status; restart may help
- **Out of memory**: The lab profile uses batch_size=8192; you can reduce it:
  Edit `configs/profiles/lab.yaml`, set `local_batch_size: 2048`
- **Import errors**: Reinstall with `pip install -e .` from the repo root
- **MLflow error**: Delete `mlflow.db` and `mlruns/`, they will be recreated

## Packing results

```powershell
# Usually called automatically by lab_train_all.ps1, but can run manually:
powershell -ExecutionPolicy Bypass -File scripts\pack_results.ps1

# Results zip is at: transfer/from_lab/results_<timestamp>.zip
# Copy this zip to USB stick
```

## Back on the laptop

```powershell
# Unpack lab results (backs up existing files first)
powershell -ExecutionPolicy Bypass -File scripts\unpack_results.ps1 `
    -ResultsZip path\to\results_<timestamp>.zip

# Verify
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Check models/serving/active.json points to the FL global model
```

## Cleanup (lab machine)

Before leaving the lab:

1. Copy `transfer/from_lab/results_*.zip` to USB
2. Optionally delete: `C:\dev\netshield-fl` (the entire repo)
3. Or just delete large files: `data/`, `.venv/`, `checkpoints/`, `mlruns/`

## Verification checklist

After unpacking on the laptop:

- [ ] `reports/tables/fl_results.csv` has all expected runs
- [ ] `reports/tables/centralized_*.json` files have `profile: "lab"`
- [ ] `models/serving/active.json` points to production ONNX
- [ ] `mlflow ui` shows experiments with correct metrics
- [ ] Production ONNX model loads and runs inference
