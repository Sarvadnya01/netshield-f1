# Demo Script — NetShield-FL (5 minutes)

## Pre-Demo Checklist

- [ ] Laptop plugged in, **Best Performance** power mode
- [ ] Docker Desktop running and warmed up (start 10+ min before)
- [ ] Notifications off (Focus Assist / Do Not Disturb)
- [ ] `docker stats` shows containers within memory limits
- [ ] Browser open with `http://localhost:8501` loaded
- [ ] Terminal ready with repo root active
- [ ] Fallback command ready in case Spark misbehaves:
  ```powershell
  # Kill spark-stream, switch to API scoring:
  docker compose --profile stream stop spark-stream
  $env:SCORING_MODE = "api"
  docker compose --profile app up -d api
  ```

## Startup

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -Rate 200
# or with API fallback scoring:
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -ScoringMode api -Rate 200
```

---

## Narration (~5 min)

### 1. Architecture Overview (30s)

> "NetShield-FL is a real-time federated intrusion detection system for IoT networks.
> It classifies network flows into 8 categories: benign traffic plus 7 attack types
> like DDoS, Mirai botnets, and reconnaissance.
>
> We use a **two-machine workflow**: rapid iteration on this dev laptop with an
> RTX 4050, and full-scale training on a lab PC with an RTX 4090. Code travels
> via git bundles; results come back as a zip with checksums. The same codebase
> runs at both scales via profile-based configuration."

**Show:** Architecture page → graphviz diagram, module mapping table.

### 2. Live SOC — Benign Traffic (45s)

> "The producer streams network flow events through Kafka. Spark Structured
> Streaming scores each event with an ONNX model in under 3 milliseconds.
> Alerts flow into this real-time dashboard.
>
> Right now we're seeing mostly benign traffic across 8 simulated organizations.
> The KPI cards update every 2 seconds — events per second, threats in the
> last minute, p95 latency, and the active model version."

**Show:** Live SOC page → KPI cards, alert table, class donut (mostly green/Benign).

### 3. Attack Injection (60s)

> "Let me simulate a DDoS attack on Organization 3."

**Action:** Sidebar → org-3, DDoS, 30s duration, 100 eps → click **Inject**.

> "Watch the alert table — DDoS predictions are now appearing for org-3 with
> high confidence. The class donut shifts from mostly benign to showing a
> significant DDoS wedge. The per-org bar chart shows org-3 spiking.
>
> This control plane uses a dedicated Kafka topic. In production, a SOC
> analyst could trigger these controls through the API."

**Show:** Alert table coloring, donut shifting, org-3 bar growing.

### 4. Federated Lab (60s)

> "Now let's look at the federated learning results."

**Show:** Federated Lab page.

> "We trained with FedAvg and FedProx under Dirichlet non-IID partitioning.
> This convergence chart shows how the global model improves over rounds.
> The horizontal lines are centralized baselines — that's the target.
>
> The client fairness boxplots show per-client F1 variation. FedProx
> with its proximal regularization term reduces this spread compared to
> FedAvg, especially under severe non-IID (alpha=0.1)."

**Show:** Round slider → animate convergence. Per-client F1 bars. Comparison chart.

> "And here's the key privacy advantage: zero bytes of raw data transmitted.
> Only model weight updates — about 1 MB per round — are shared with the
> server. Each client's data stays on their device."

**Show:** Privacy card → 0 bytes raw data vs N MB model updates.

### 5. Experiments + Model (45s)

> "The experiments page shows all our results. Per-class F1 heatmaps
> reveal where federated learning struggles — minority classes like Web
> and BruteForce under non-IID conditions. Interestingly, removing the
> inter-arrival time feature actually improves MLP performance by 11%."

**Show:** Experiments page → results table, per-class heatmap, IAT ablation.

> "The ONNX model serves predictions in 0.2ms per batch of 256 — well
> within real-time requirements."

### 6. Wrap-up (30s)

> "To summarize: NetShield-FL demonstrates that federated learning can
> produce viable intrusion detection models while keeping data local,
> and serve them in real-time through a production-grade streaming pipeline."

---

## Likely Faculty Questions

**Q: Why build a custom FL engine instead of using Flower or PySyft?**
> For pedagogical value and full control. Our engine is ~400 lines of code,
> supports resume/checkpoint, and makes the FedAvg/FedProx math explicit.
> No framework magic to debug during the demo.

**Q: Why macro-F1 instead of accuracy?**
> The dataset has significant class imbalance. Accuracy would be dominated
> by the majority class. Macro-F1 weights each class equally, exposing
> weaknesses on minority attack types.

**Q: Why Spark Structured Streaming instead of Flink or a simpler consumer?**
> Spark provides exactly-once semantics, watermark-based windowing for
> metrics aggregation, and mapInPandas for zero-copy ONNX inference. We
> also have a fallback API-mode scorer if Spark misbehaves during the demo.

**Q: What does "non-IID" mean in this context?**
> Each simulated client sees a different distribution of attack types,
> controlled by a Dirichlet parameter alpha. Low alpha (0.1) means clients
> specialize in different attacks — realistic for IoT deployments where
> different sites face different threat profiles.

**Q: What did you learn from the IAT ablation?**
> Inter-arrival time in aggregated flow records is noisy and actually hurts
> classification. Removing it improved macro-F1 by ~11%. This highlights
> that automated feature engineering still needs domain validation.

**Q: Why train on a separate GPU machine?**
> The dev laptop has 6 GB VRAM — sufficient for smoke tests but not for
> the full CICIoT2023 dataset (~46M rows) and a 20-round FL grid with
> 3 seeds. The lab's RTX 4090 (24 GB) handles full-scale training, while
> the laptop runs the streaming pipeline and dashboard.
