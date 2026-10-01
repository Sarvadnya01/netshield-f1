# NetShield-FL — context for Claude Code

## Project
Scalable ML & Big Data Analytics mini-project. Real-time federated IoT intrusion
detection on CICIoT2023.
Research question: under non-IID client data (Dirichlet label skew α ∈ {0.1, 0.5, IID}),
how close do FedAvg and FedProx get to centralized training on 8-class IoT attack
classification, and can the federated global model serve live streamed traffic at
production latency?

## Two-machine workflow (IMPORTANT)
- Claude Code runs ONLY on the dev laptop: Windows 11, RTX 4050 Laptop (6 GB VRAM),
  16 GB RAM, Docker Desktop with WSL2 capped at 8 GB. Repo path C:\dev\netshield-fl.
- Full-scale training runs on a college lab PC (i9-13950HX, RTX 4090, Windows, no Docker,
  possibly no admin rights, no GitHub credentials). Same repo path C:\dev\netshield-fl.
- Profiles: configs/*.yaml are base configs, deep-merged with
  configs/profiles/{laptop,lab}.yaml selected by env NETSHIELD_PROFILE (default "laptop").
- On the laptop, NEVER launch full-scale training or the full FL grid. Run only
  smoke-scale with the laptop profile, which must finish in minutes.
- Lab code arrives via `git bundle`; lab results return via a zip produced by
  scripts/pack_results.ps1 and are committed from the laptop. The lab never pushes.
- MLflow: laptop uses sqlite:///mlflow_laptop.db (scratch); the lab writes the canonical
  sqlite:///mlflow.db + mlruns/ that come back in the results bundle.
- The model used by streaming/API/dashboard is chosen by models/serving/active.json
  (smoke model first, FL global model after the lab run). Never hardcode a model filename.

## Data flow
CSVs → Spark batch ETL (Docker, laptop) → Parquet splits + stream holdout
→ GPU training (smoke: laptop, full: lab): XGBoost & MLP baselines; custom FL engine
  (FedAvg/FedProx) → MLflow → ONNX export → models/serving/
→ producer (host) → Kafka iot.flows → Spark Structured Streaming + ONNX (Docker)
→ iot.alerts + iot.metrics → FastAPI → Streamlit. iot.control carries attack-injection commands.

## Environment
- Host venv .venv, Python 3.11. PyTorch CUDA 12.x wheels installed separately.
- Spark 3.5.x + Java 17 ONLY inside docker/spark, with numpy<2 there. Never run Spark on host Windows.
- Kafka apache/kafka:3.9.1 KRaft. Containers use kafka:9092; host uses localhost:9094.
- Docker services must declare memory limits that fit inside the 8 GB WSL2 budget.

## Conventions
- Package src/netshield (`pip install -e .`). pathlib only; all paths via
  netshield.common.config. No absolute paths in code or committed configs.
- Class order fixed by netshield.common.labels.CLASS_NAMES (Benign, DDoS, DoS, Mirai,
  Recon, Spoofing, Web, BruteForce).
- Feature preprocessing exists ONLY in netshield.common.preprocess; stats in
  models/serving/feature_stats.json.
- Message schemas: netshield.common.schemas (Pydantic v2) + docs/data_contracts.md.
- Models use LayerNorm, never BatchNorm.
- Long jobs must be resumable (skip finished work, checkpoint progress) and log to logs/.
- Every experiment: fixed seed, logged to MLflow, JSON summary under reports/.
- Primary metric macro-F1; always also report per-class metrics.
- Type hints, docstrings, `logging`, pytest for core logic. PowerShell scripts only (no bash/make).

## Git rules
- Never commit data/, transfer/, logs/, mlruns/, *.db, checkpoints/, .venv, or files > 5 MB
  (exception: models/serving/*.onnx if < 5 MB).
- When a phase passes acceptance: pytest → `git add -A` → conventional commit →
  `git pull --rebase origin main` → `git push origin HEAD`. Update the checklist below.

## Phase status
- [x] P0 Bootstrap (profiles, contracts, infra)
- [x] P1 Spark ETL (full run on laptop)
- [x] P2 Centralized baselines + ONNX (code + laptop smoke)
- [x] P3 Federated engine (code + laptop smoke)
- [x] PL1 Lab migration tooling
- [ ] — LAB SESSION 1 (manual, docs/lab_runbook.md)
- [ ] P4 Kafka + Spark Structured Streaming
- [ ] P5A FastAPI
- [ ] P5B Streamlit dashboard
- [ ] PL2 Integrate lab results
- [ ] P6 Evaluation + report assets
- [ ] P7 Integration + demo hardening