# 🕵️ Credit Card Fraud Detection

![PR-AUC](https://img.shields.io/badge/PR--AUC-0.9284-brightgreen?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=for-the-badge&logo=fastapi)
![Tests](https://img.shields.io/badge/Tests-17%20Passing-brightgreen?style=for-the-badge)
![CI](https://github.com/mallapuabhiraj/fraud-detection-catboost/actions/workflows/tests.yml/badge.svg)

> *"Finding 6 needles in a haystack of 994 — without setting the haystack on fire."*

End-to-end fraud detection system — CatBoost model with **0.93 PR-AUC** on a brutally imbalanced dataset (0.6% fraud rate), served via a production-ready FastAPI with PostgreSQL prediction logging, 17 unit tests, and CI/CD. Every decision in this project — dataset, metric, model, threshold — has a documented reason behind it.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Test PR-AUC** | **0.9284** |
| Test ROC-AUC | 0.9986 |
| Fraud Recall | 0.89 |
| Precision | 0.80 |
| Val → Test Gap | 0.028 |

> Evaluated on held-out `fraudTest.csv` — never seen during training or tuning. Not even once. We mean it.

---

## 🚨 The Problem

Credit card fraud costs **billions annually**. But here's the part nobody puts in their project description:

- Only **0.6% of transactions are fraudulent**
- A model that classifies *everything* as legitimate scores **99.4% accuracy**
- That model would be completely, catastrophically useless in production

Standard accuracy is not a metric here — it's a lie. This project optimizes for what actually matters: **catching fraud without burying analysts in false alarms.** That means PR-AUC, cost-sensitive thresholds, and real-world generalization testing. Not leaderboard climbing.

---

## 📦 Dataset Selection — Why Sparkov, Not the Standard Kaggle Dataset

This is where most fraud detection projects make their first quiet mistake — and nobody mentions it.

### The Path of Least Resistance — Anonymized Datasets (IEEE-CIS style)

The most popular fraud dataset on Kaggle comes pre-anonymized: 28 features named `V1` through `V28`, already PCA-transformed and scaled before you touch them. Plug it into any gradient boosting model, tune for two hours, watch ROC-AUC climb to 0.99.

The tradeoff nobody talks about:

- `V14 > 0.3` flags fraud — but **why?** Nobody knows. Not you, not the model, not the risk team asking for an explanation at 3am when a false alarm freezes a customer's card
- No `lat` / `long` → can't compute `distance_km` between cardholder and merchant
- No merchant location → impossible travel detection is, ironically, impossible
- No raw timestamps → no behavioral velocity, no transaction history features
- No transaction context → dormant card reactivation, card testing patterns — all invisible

The model becomes a black box sitting on top of another black box. The numbers look impressive. The understanding is zero. And every other project in the search results looks identical.

### The Harder, More Interesting Route — Sparkov Simulation

Sparkov generates raw, realistic credit card transactions — the kind that actually flow through payment networks before anyone anonymizes them.

```
trans_date_trans_time · cc_num · merchant · category · amt
gender · city · state · lat · long · city_pop
job · dob · merch_lat · merch_long
```

Every field is interpretable. Every field is an engineering opportunity.

| Dimension | Anonymized (V1–V28) | Sparkov (this project) |
|---|---|---|
| Feature interpretability | ❌ PCA-transformed, opaque | ✅ Real transaction fields |
| Feature engineering headroom | ❌ Nothing left to build | ✅ 40+ hand-crafted signals |
| Business explainability | ❌ `V14` means nothing to anyone | ✅ Every feature has a documented reason |
| Temporal structure | ❌ No meaningful ordering | ✅ Full chronological splits |
| Distribution drift testing | ❌ Not possible | ✅ Tested across fraud rates |
| Preprocessing ownership | ❌ Already done upstream | ✅ Built from scratch |

Choosing Sparkov is what made 40+ domain-driven features possible. It's also what made the generalization failure detectable — and fixable. A dataset that hides its structure hides your mistakes right along with it.

> *"The anonymized dataset is a perfectly valid benchmark. This project needed a foundation to actually build on."*

---

## 🌍 Generalization — Out-of-Time Drift Test

Choosing the right dataset surfaces problems that clean benchmarks quietly bury. This one surfaced fast.

The base model trained on `fraudTrain.csv` at **0.6% fraud rate**. Tested against new, unseen Sparkov transactions (Jan–Jun 2023) at **2.2% fraud rate** — it collapsed. The model had simply never been trained to recognize fraud at that frequency.

The fix wasn't volume. It was distribution: retrain on a dataset whose **fraud rate matched the real-world test distribution**.

| | Base Model | Retrained Model |
|---|---|---|
| Training Fraud Rate | 0.6% | ~2% |
| Test Fraud Rate | 2.2% | 2.2% |
| Out-of-Time PR-AUC | 0.6471 ❌ | 0.9997 ✅ |
| Improvement | — | **+35.26%** |

> A model that only sees 0.6% fraud during training will struggle when fraud becomes 4× more common in production. Distribution matching is not a nice-to-have — it is the job.

---

## 🏗️ Architecture

```
Client Request (JSON)
        ↓
  FastAPI  (api/main.py)
        ↓                    ↓
CatBoost Inference     PostgreSQL (Neon)
   + Risk Scoring       Prediction Logging
        ↓
  JSON Response
```

---

## 🧠 Key Technical Decisions

### Why PR-AUC Over ROC-AUC
At 0.6% fraud rate, ROC-AUC is actively misleading — a model that flags nothing as fraud still scores 0.99. PR-AUC measures precision and recall directly on the minority class. It is the only metric here that cannot be gamed by predicting the majority.

### Why CatBoost Over XGBoost
The first model was XGBoost with Leave-One-Out target encoding. It achieved **0.67 PR-AUC** on the test set with a **0.11 val→test gap** — strong validation numbers masking poor generalization.

The root cause: LOO encoding was leaking future category-level fraud statistics into training. The model memorized the encoding artifact, not the fraud signal.

Switching to CatBoost's native ordered target statistics — which encode each row using only transactions that chronologically preceded it — eliminated the leak:

- Val → Test gap: **0.11 → 0.028**
- Test PR-AUC: **0.67 → 0.93**

### Why Strict Time-Based Splits
Fraud patterns are temporal. A transaction from December cannot inform a model evaluating January. Random splits violate this — they leak future signal into training and produce validation scores that won't survive deployment. Every split in this project is strictly chronological, no exceptions.

### Why Threshold 0.830
The default 0.5 optimizes for balanced accuracy — not business cost. The original threshold of 0.862 was derived from an assumed FP:FN ratio of 1:6 with no external basis.

After independently reviewing four industry sources — **LexisNexis True Cost of Fraud 2024**, **Ravelin**, **MIT LIDS**, and **Chargeflow** chargeback data — two separate cost frameworks independently converged on **0.830** as the optimal operating point for a bank/card-issuer context. Total operational cost at this threshold is **7.6% lower** than at 0.862.

The threshold has a citation. Not just a config value.

---

## ⚡ Feature Engineering — Top Signals

No single feature dominates. The top feature (`category`) accounts for 14.57% of importance, distributed across 55+ engineered signals. No leakage detected.

| Feature | Signal |
|---------|--------|
| `category` + `amt_zscore_category` | 28% combined — fraud in grocery is structurally different from fraud in online shopping |
| `trans_num_per_card` | Sudden transaction volume spikes are a reliable early warning |
| `hour` | Fraudulent activity concentrates heavily in late-night hours |
| `is_impossible_travel` | Speed > 500 km/h between consecutive transactions = stolen card, not teleportation |
| `is_dormant_reactivation` | 7+ day gap followed by an unusually large transaction — a classic pattern |
| `merchant_first_use` | First appearance of a card at a new merchant, combined with high spend |
| `is_rapid_succession` | 3+ transactions within 5 minutes = card testing before the big fraudulent charge |

---

## 🔍 ML Pipeline

```
Raw Data — fraudTrain.csv (1.2M rows, 0.6% fraud rate)
        ↓
Chronological Train / Validation Split (80 / 20)
        ↓
Feature Engineering — 40+ domain-driven signals
        ↓
CatBoost with Native Ordered Categorical Encoding
        ↓
Optuna Hyperparameter Optimization
(20 trials · TPE sampler · PR-AUC objective)
        ↓
Cost-Sensitive Threshold Calibration
(LexisNexis 2024 · Ravelin · MIT LIDS · Chargeflow)
        ↓
Final Holdout Evaluation — fraudTest.csv (555K rows, never touched before)
        ↓
Out-of-Time Generalization Test — Sparkov Jan–Jun 2023 (2.2% fraud rate)
```

---

## 🎛️ Best Hyperparameters

Found via Optuna TPE search over 20 trials. `l2_leaf_reg` (Optuna importance: 0.28) and `bagging_temperature` (0.25) carry most of the regularization weight.

```python
{
    'learning_rate':       0.0375,
    'depth':               10,
    'l2_leaf_reg':         7.976,    # Optuna importance: 0.28 — primary regularizer
    'scale_pos_weight':    243.95,
    'bagging_temperature': 0.895,    # Optuna importance: 0.25 — stochastic regularization
    'random_strength':     1.995,
    'border_count':        240,
}
```

---

## 🌐 API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status, model status, database connectivity |
| `POST` | `/predict` | Single transaction — returns probability, prediction, risk level |
| `POST` | `/predict-batch` | Batch of transactions — same response per item |
| `GET` | `/predictions/history` | Last 50 logged predictions from PostgreSQL |

### Sample Request

```json
{
    "trans_date_trans_time": "2024-01-15 14:32:00",
    "cc_num": 4532015112830366,
    "merchant": "fraud_Rippin, Kub and Mann",
    "category": "shopping_net",
    "amt": 149.62,
    "gender": "M",
    "city": "Anytown",
    "state": "TX",
    "zip": 75001,
    "lat": 33.9659,
    "long": -80.9355,
    "city_pop": 149180,
    "job": "Software Engineer",
    "dob": "1990-03-15",
    "merch_lat": 33.986391,
    "merch_long": -81.200714
}
```

### Sample Response

```json
{
    "fraud_probability": 0.995,
    "prediction": "fraud",
    "risk_level": "Critical Risk"
}
```

### Risk Level Bands

Calibrated against the model's probability distribution. 84% of all confirmed fraud scores at Critical Risk.

| Level | Range | Interpretation |
|-------|-------|----------------|
| 🟢 Low Risk | < 0.830 | Below decision threshold — transaction looks legitimate |
| 🟡 Elevated Risk | 0.830 – 0.90 | Above threshold — moderate model confidence |
| 🔴 High Risk | 0.90 – 0.99 | High-confidence fraud signal |
| 🚨 Critical Risk | ≥ 0.99 | Near-certain fraud — where 84% of real fraud lands |

---

## 🗂️ Project Structure

```
fraud-detection-catboost/
│
├── .github/
│   └── workflows/
│       └── tests.yml               # CI/CD — triggers on every push to main
│
├── api/
│   ├── config.py                   # FRAUD_THRESHOLD, model paths, categorical columns
│   ├── database.py                 # SQLAlchemy ORM + Neon PostgreSQL connection
│   ├── main.py                     # FastAPI application and endpoint definitions
│   ├── model_services.py           # Inference logic and risk band assignment
│   └── schemas.py                  # Pydantic request and response models
│
├── images/
│   ├── cost_threshold.png
│   ├── cost_sensitivity_research.png
│   ├── feature_importance.png
│   ├── probability_distribution.png
│   └── pr_curve.png
│
├── models/                         # gitignored — see Quick Start step 4
│   ├── cb_model.cbm
│   └── preprocess_pipe.joblib
│
├── notebooks/
│   └── fraud_detection.ipynb       # Full pipeline: EDA → training → evaluation
│
├── src/
│   └── preprocess.py               # All 40+ feature engineering transformations
│
├── tests/
│   └── test_api.py                 # 17 pytest unit tests
│
├── config.json                     # Optuna best hyperparameters
├── requirements.txt
└── README.md
```

---

## 💻 Quick Start

### Prerequisites

- Python 3.10+
- Git
- A [Neon](https://neon.tech) free account for PostgreSQL
- Model files — `cb_model.cbm` and `preprocess_pipe.joblib` (see step 4)

### 1. Clone

```bash
git clone https://github.com/mallapuabhiraj/fraud-detection-catboost.git
cd fraud-detection-catboost
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

`(venv)` should appear in your terminal prompt. If it doesn't — stop here and fix this first.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model files

The `models/` folder is gitignored — files are too large for GitHub. Create it and copy the model files in:

```bash
# Windows
mkdir models

# Mac / Linux
mkdir -p models
```

Copy `cb_model.cbm` and `preprocess_pipe.joblib` into `models/`.

### 5. Configure environment

Create a `.env` file in the project root:

```bash
# Mac / Linux
echo "DATABASE_URL=your_neon_connection_string" > .env

# Windows (PowerShell)
Add-Content .env "DATABASE_URL=your_neon_connection_string"
```

### 6. Run

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI. If it loads — you're done. 🎉

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

```
17 passed
```

---

## 📋 Data Reference

Source: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Sparkov simulation). Not included in this repository.

| Split | Rows | Fraud Rate | Purpose |
|-------|------|------------|---------|
| `fraudTrain.csv` | ~1.2M | ~0.6% | Training — base model |
| `fraudTest.csv` | ~555K | ~0.6% | Final holdout evaluation |
| New Sparkov Jan–Jun 2023 | ~185K | ~2.2% | Out-of-time generalization test |

---

## 🛠️ Skills Demonstrated

`CatBoost` `FastAPI` `PostgreSQL` `SQLAlchemy` `pytest` `GitHub Actions` `CI/CD`
`Optuna` `Feature Engineering` `Imbalanced Classification` `Temporal Validation`
`Cost-Sensitive Thresholding` `Hyperparameter Tuning` `REST API Design`
`Distribution Drift` `Python` `Scikit-learn`
