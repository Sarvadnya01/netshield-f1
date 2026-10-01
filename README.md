# NetShield-FL

Real-time federated IoT intrusion detection on CICIoT2023.

8-class classification (Benign, DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce)
using FedAvg/FedProx under non-IID label skew, with live Kafka streaming inference.

## Quickstart (Laptop)

### Prerequisites

- Python 3.11, Git, Docker Desktop (WSL2 backend)
- NVIDIA GPU with CUDA 12.x drivers
- WSL2 memory cap: copy `docs/wslconfig.example` to `%UserProfile%\.wslconfig` and restart WSL

```
[wsl2]
memory=8GB
processors=8
swap=8GB
```

### 1. Setup Environment

```powershell
.\scripts\setup_env.ps1 -Profile laptop
.\.venv\Scripts\Activate.ps1
python scripts\verify_env.py
```

### 2. Start Kafka

```powershell
docker compose up -d kafka kafka-init
# Wait for healthy, then verify topics:
docker exec netshield-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
```

### 3. Run Tests

```powershell
pytest
```

### 4. Build Spark Image

```powershell
docker compose build spark-etl
```

## Project Structure

```
configs/            Base YAML configs + profiles/ (laptop, lab)
docker/             Dockerfiles for spark, api, dashboard
docs/               Architecture, data contracts, lab runbook
models/serving/     ONNX models + feature_stats.json + active.json
reports/            Experiment results, figures, tables
scripts/            PowerShell setup/run scripts
src/netshield/      Python package
  common/           Config, labels, schemas, preprocessing, hardware
  etl/              Spark batch ETL
  models/           MLP, XGBoost, training, evaluation, ONNX export
  federated/        FL engine (FedAvg, FedProx)
  streaming/        Kafka producer, Spark Structured Streaming
  api/              FastAPI backend
  dashboard/        Streamlit frontend
tests/              pytest test suite
```

## Two-Machine Workflow

- **Laptop** (dev): smoke-scale training, Docker services, streaming demo
- **Lab PC** (RTX 4090): full-scale training & FL grid via `git bundle`

See `docs/lab_runbook.md` for lab session instructions.
