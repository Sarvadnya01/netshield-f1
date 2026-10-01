# Data Contracts

All Kafka message schemas use **epoch-millisecond** timestamps.
Pydantic v2 models live in `netshield.common.schemas`.

## Kafka Topics

### `iot.flows` — Raw Flow Events

| Field       | Type              | Description                     |
|-------------|-------------------|---------------------------------|
| event_id    | str               | UUID for the flow               |
| ts_event    | int (epoch-ms)    | When the flow was captured      |
| org_id      | str               | Organization/tenant ID          |
| device_id   | str               | Source device identifier        |
| features    | dict[str, float]  | Numeric feature vector          |
| true_label  | str \| null       | Ground-truth label (if known)   |

### `iot.alerts` — Scored Alerts

| Field         | Type              | Description                        |
|---------------|-------------------|------------------------------------|
| event_id      | str               | Matches the flow event_id          |
| ts_event      | int (epoch-ms)    | Original flow timestamp            |
| ts_scored     | int (epoch-ms)    | When scoring completed             |
| latency_ms    | float             | Scoring latency in milliseconds    |
| org_id        | str               | Organization/tenant ID             |
| device_id     | str               | Source device identifier           |
| pred_class    | str               | Predicted class name               |
| confidence    | float             | Confidence score (max probability) |
| probs         | dict[str, float]  | Per-class probabilities            |
| is_attack     | bool              | True if pred_class != "Benign"     |
| true_label    | str \| null       | Ground-truth label (if known)      |
| model_version | str               | Model identifier                   |

### `iot.metrics` — Aggregated Metrics

| Field        | Type              | Description                      |
|--------------|-------------------|----------------------------------|
| window_start | int (epoch-ms)    | Window start time                |
| window_end   | int (epoch-ms)    | Window end time                  |
| org_id       | str               | Organization/tenant ID           |
| total        | int               | Total flows in window            |
| attacks      | int               | Attack flows in window           |
| by_class     | dict[str, int]    | Per-class counts                 |

### `iot.control` — Control Commands

| Field        | Type   | Description                              |
|--------------|--------|------------------------------------------|
| command      | str    | "inject" or "stop"                       |
| org_id       | str    | Target organization                      |
| attack_class | str    | Attack class to inject                   |
| duration_s   | float  | Duration of injection in seconds         |
| rate_eps     | float  | Events per second for injection          |

## Serving Metadata

### `models/serving/active.json` — Active Model

| Field      | Type   | Description                           |
|------------|--------|---------------------------------------|
| model_file | str    | ONNX filename in models/serving/      |
| model_card | dict   | Training metadata (metrics, params)   |
| kind       | str    | "smoke" or "production"               |
| updated_at | str    | ISO-8601 timestamp                    |

### `models/serving/feature_stats.json` — Feature Stats

| Field         | Type        | Description                        |
|---------------|-------------|------------------------------------|
| feature_names | list[str]   | Ordered feature column names       |
| mean          | list[float] | Per-feature mean after log1p       |
| std           | list[float] | Per-feature std after log1p        |

## Class Order

Fixed in `netshield.common.labels.CLASS_NAMES`:

1. Benign
2. DDoS
3. DoS
4. Mirai
5. Recon
6. Spoofing
7. Web
8. BruteForce
