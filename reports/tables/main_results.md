| Method            | Alpha   |   Macro-F1 |   Accuracy | Client-F1 Std   | Rounds to 95%   | MB Communicated   | Profile   |
|:------------------|:--------|-----------:|-----------:|:----------------|:----------------|:------------------|:----------|
| MLP               | -       |     0.1108 |     0.123  | -               | -               | -                 | laptop    |
| MLP               | -       |     0.1001 |     0.1221 | -               | -               | -                 | laptop    |
| XGB               | -       |     0.1257 |     0.1276 | -               | -               | -                 | laptop    |
| FEDAVG            | 0.5     |     0.0684 |     0.1226 | 0.0211          | not reached     | 3.12              | laptop    |
| FedProx (mu=0.01) | 0.5     |     0.0684 |     0.1226 | 0.0211          | not reached     | 3.12              | laptop    |