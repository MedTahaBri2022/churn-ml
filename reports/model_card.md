# Model card: customer churn classifier

- **Model**: `gradient_boosting` {'learning_rate': 0.05, 'max_depth': 3}
- **Trained**: 2026-10-01T20:09:28+00:00 (git `None`)
- **Data**: 5634 training rows, 1409 test rows, churn rate 26.5%, sha256 `16320c9c1ec7`

## Model selection (5-fold cross-validation on the training split)

| Model | ROC-AUC | PR-AUC | Best parameters |
| --- | ---: | ---: | --- |
| baseline_prior | 0.500 | 0.265 | - |
| logistic_regression | 0.848 | 0.661 | {'C': 0.1} |
| random_forest | 0.848 | 0.664 | {'max_depth': 10, 'min_samples_leaf': 20} |
| gradient_boosting | 0.849 | 0.668 | {'learning_rate': 0.05, 'max_depth': 3} |

## Hold-out test set (used once)

- ROC-AUC **0.846**, PR-AUC **0.662**, Brier score 0.135
- Decision threshold **0.19**, chosen on out-of-fold training predictions to minimize cost (a missed churner costs 5x a wasted retention offer)
- At that threshold: precision 0.460, recall 0.853, F1 0.598
- At the default 0.50: precision 0.681, recall 0.519

| | Predicted stay | Predicted churn |
| --- | ---: | ---: |
| **Stayed** | 661 | 374 |
| **Churned** | 55 | 319 |

## Most influential features (permutation importance, drop in ROC-AUC)

- `Contract`: 0.0819
- `tenure`: 0.0663
- `InternetService`: 0.0095
- `OnlineSecurity`: 0.0054
- `TechSupport`: 0.0053
- `PaymentMethod`: 0.0032
- `MonthlyCharges`: 0.0025
- `StreamingMovies`: 0.0016

## Intended use and limits

- Ranks customers by churn risk to prioritize retention actions; not a basis for decisions that affect a customer's rights or pricing.
- Trained on one public snapshot of a telecom operator: no time dimension, so the model cannot be validated on a later period, which is how it would be used in production.
- `gender` and `SeniorCitizen` are inputs; fairness across those groups has not been audited.
