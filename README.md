# NetShield-FL

Real-time federated IoT intrusion detection on CICIoT2023.

8-class classification (Benign, DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce)
using FedAvg/FedProx under non-IID label skew, with live Kafka streaming inference.

## Prerequisites

- **Python 3.11**, Git, **Docker Desktop** (WSL2 backend)
- NVIDIA GPU with CUDA 12.x drivers (for training; inference uses ONNX CPU)
- ~20 GB disk for Docker images, data, and models

### WSL2 Memory Cap (required)

Docker Desktop runs inside WSL2. Cap its memory to avoid starving the host:

```
# Save as %UserProfile%\.wslconfig, then: wsl --shutdown && restart Docker Desktop
[wsl2]
memory=8GB
processors=8
swap=8GB
```

A template is at `docs/wslconfig.example`.

## Quick Start

### 1. Setup Environment

```powershell
.\scripts\setup_env.ps1 -Profile laptop
.\.venv\Scripts\Activate.ps1
python scripts\verify_env.py
```

### 2. Download Dataset

Download CICIoT2023 CSVs from [the dataset page](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
and place them in `data/raw/ciciot2023/`.

### 3. Run Spark ETL

```powershell
docker compose build spark-etl
docker compose --profile etl up         # writes data/splits/ + data/stream/
```

### 4. Smoke Training (laptop)

```powershell
python -m netshield.models.train_centralized   # MLP + XGBoost baselines
python -m netshield.federated.run_experiment    # FedAvg + FedProx smoke grid
```

### 5. Run Tests

```powershell
pytest
```

### 6. Start Demo

```powershell
# Normal (Spark scoring):
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1

# Fallback (API scoring, no Spark):
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -ScoringMode api

# Stop everything:
powershell -ExecutionPolicy Bypass -File scripts\stop_demo.ps1
```

- **Dashboard:** http://localhost:8501
- **API docs:** http://localhost:8000/docs
- **Kafka UI:** `docker compose --profile tools up -d kafka-ui` → http://localhost:8080

### Mock Mode (no Docker needed)

For development or on machines without Docker:

```powershell
$env:MOCK_MODE = "1"
uvicorn netshield.api.main:app --port 8000           # terminal 1
streamlit run src/netshield/dashboard/app.py          # terminal 2
```

## Project Structure

```
configs/            Base YAML configs + profiles/ (laptop, lab)
docker/             Dockerfiles for spark, api, dashboard
docs/               Architecture, data contracts, lab runbook, demo script
models/serving/     ONNX models + feature_stats.json + active.json
reports/            Experiment results, figures, tables, final report
scripts/            PowerShell setup/run/demo scripts
src/netshield/      Python package
  common/           Config, labels, schemas, preprocessing, hardware
  etl/              Spark batch ETL
  models/           MLP, XGBoost, training, evaluation, ONNX export
  federated/        FL engine (FedAvg, FedProx, partitioning)
  streaming/        Kafka producer, Spark Structured Streaming, benchmark
  api/              FastAPI backend (live alerts, control, FL results, predict)
  dashboard/        Streamlit SOC dashboard (4 pages)
tests/              pytest test suite
```

## Two-Machine Workflow

| Machine | Hardware | Role |
|---------|----------|------|
| Dev laptop | RTX 4050 (6 GB), 16 GB RAM | Smoke training, Docker services, streaming, dashboard |
| Lab PC | RTX 4090 (24 GB), i9-13950HX | Full-scale training & FL grid |

Code transfers via `git bundle`; results return as a zip with SHA-256 checksums.
Profile-based config (`configs/profiles/laptop.yaml` vs `lab.yaml`) ensures the
same code runs at both scales.

See `docs/lab_runbook.md` for lab session instructions.

## Troubleshooting

**Docker Desktop won't start / "unable to start"**
- Restart Docker Desktop from the system tray
- Run `wsl --shutdown` then restart Docker Desktop
- Check Windows Features: WSL2 and Virtual Machine Platform must be enabled

**Port conflicts (8000, 8501, 9094)**
```powershell
netstat -ano | findstr :8000    # find the PID using the port
taskkill /PID <pid> /F          # kill it
```

**Kafka connection refused on localhost:9094**
- Ensure Kafka is running: `docker compose ps kafka`
- Check the container is healthy (takes ~30s after start)
- Verify the EXTERNAL listener is on port 9094: `docker compose logs kafka | Select-String EXTERNAL`

**Docker memory errors / OOM**
- Ensure `.wslconfig` limits WSL2 to 8 GB
- Run `wsl --shutdown` and restart Docker Desktop after changing `.wslconfig`
- Check with `docker stats` — total should stay under 6 GB

**OneDrive paths (spaces in path)**
- Clone the repo to a short path like `C:\dev\netshield-fl`
- Do NOT clone into OneDrive, Desktop, or Documents folders

**Spark streaming not producing alerts**
- Use API fallback: `scripts\start_demo.ps1 -ScoringMode api`
- Check Spark logs: `docker compose --profile stream logs spark-stream`
- Clear checkpoints: `scripts\restart_stream.ps1 -ClearCheckpoints`
