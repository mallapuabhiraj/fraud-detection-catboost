# 🕵️ Credit Card Fraud Detection

![PR-AUC](https://img.shields.io/badge/PR--AUC-0.9284-brightgreen?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=for-the-badge&logo=fastapi)
![Tests](https://img.shields.io/badge/Tests-17%20Passing-brightgreen?style=for-the-badge)
![CI](https://github.com/mallapuabhiraj/fraud-detection-catboost/actions/workflows/tests.yml/badge.svg)

> *"Finding 6 needles in a haystack of 994 — without setting the haystack on fire."*

End-to-end fraud detection system — CatBoost model with **0.93 PR-AUC** on a brutally imbalanced dataset (0.6% fraud rate), served via a production-ready FastAPI with PostgreSQL logging, unit tests, and CI/CD.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Test PR-AUC** | **0.9284** |
| Test ROC-AUC | 0.9986 |
| Fraud Recall | 0.89 |
| Precision | 0.80 |
| Val→Test Gap | 0.028 |

> Evaluated on held-out `fraudTest.csv` — never seen during training or tuning. Not even once. We're serious about that.

---

## 🌍 Generalization — Out-of-Time Drift Test

The base model was trained on `fraudTrain.csv` with a **0.6% fraud rate**.
When tested against new unseen Sparkov data (Jan–Jun 2023) with a **2.2% fraud rate** — it collapsed. The model had never seen fraud this frequent.

The fix was simple but important: retrain on a larger, more diverse dataset that naturally contains a **~2% fraud rate** — closer to what the test data looks like.

| | Base Model | Retrained Model |
|---|---|---|
| **Training Fraud Rate** | 0.6% | ~2% |
| **Test Fraud Rate** | 2.2% | 2.2% |
| **Out-of-Time PR-AUC** | 0.6471 ❌ | 0.9997 ✅ |
| **Improvement** | — | **+35.26%** |

> The key insight — it wasn't just *more data* that helped. It was training on data whose **fraud rate matched the real-world distribution**. A model that only sees 0.6% fraud during training struggles when fraud suddenly becomes 4x more common in production.

---

## 🏗️ Architecture

```
Client (JSON)
      ↓
FastAPI (api/main.py)
      ↓              ↓
CatBoost Model    PostgreSQL (Neon)
(prediction)      (prediction logging)
```

---

## 🚨 The Problem

Credit card fraud costs **billions annually**. But here's the twist nobody talks about:

- Only **0.6% of transactions are fraudulent**
- A model that predicts *"everything is legitimate"* scores **99.4% accuracy**
- That model is completely useless

Standard accuracy metrics lie here. This project optimizes for what actually matters — **finding fraud without drowning analysts in false alarms.**

---

## 🧠 Key Technical Decisions

### Why PR-AUC over ROC-AUC
At 0.6% fraud rate, ROC-AUC is misleading — a model predicting everything as legitimate still scores 0.99. PR-AUC directly measures the precision-recall tradeoff on the minority class. It's the only honest metric here.

### Why CatBoost over XGBoost
XGBoost with Leave-One-Out encoding achieved **0.67 test PR-AUC** with a painful **0.11 val→test gap**. Switching to CatBoost's native ordered target statistics:
- Eliminated encoding drift between train and test periods
- Reduced val→test gap from **0.11 → 0.028**
- Improved test PR-AUC from **0.67 → 0.93**

### Why Time-Based Split
Fraud data is temporal. Random splits leak future transaction patterns into training — inflating scores without improving real detection. Every single evaluation in this project uses strict temporal ordering. No cheating.

### Why Threshold 0.830
The original threshold was derived assuming FP=1, FN=6 — a made-up ratio with zero research backing. After analyzing real industry cost data across **LexisNexis True Cost of Fraud 2024**, **Ravelin**, **MIT LIDS**, and **Chargeflow** chargeback data — two independent cost frameworks converged on **0.830** as optimal for a bank/card issuer use case, reducing total business cost by **7.6%** over the original threshold.

---

## ⚡ Top Features

Feature importance validated — no single feature dominates (top feature `category` = 14.57%, distributed across 55+ features). No leakage.

| Feature | Why It Works |
|---------|--------------|
| `category` + `amt_zscore_category` | 28% combined — fraud in grocery looks nothing like fraud in shopping |
| `trans_num_per_card` | New or suddenly hyperactive cards are suspicious |
| `hour` | Fraudsters love the 2 AM shift |
| `is_impossible_travel` | Speed > 500 km/h between transactions = stolen card, not teleportation |
| `is_dormant_reactivation` | >7 day gap + high amount = someone found grandma's old card |
| `merchant_first_use` | First time card seen at merchant + high amount = classic fraud setup |
| `is_rapid_succession` | 3 transactions in 5 minutes = card testing before the big hit |

---

## 🔍 ML Pipeline

```
Raw Data (fraudTrain.csv — 1.2M rows, 0.6% fraud)
        ↓
Time-Based Split (80/20) — no peeking at the future
        ↓
Feature Engineering (40+ features)
        ↓
CatBoost with Native Categorical Encoding
        ↓
Optuna Hyperparameter Tuning (20 trials, TPE sampler)
        ↓
Cost-Sensitive Threshold Tuning
(LexisNexis 2024 + MIT LIDS — not just vibes)
        ↓
Final Evaluation on fraudTest.csv
        ↓
Out-of-Time Drift Test on new Sparkov data (2.2% fraud rate)
```

---

## 🎛️ Best Hyperparameters

```python
{
    'learning_rate':       0.0375,
    'depth':               10,
    'l2_leaf_reg':         7.976,   # most important — 0.28 Optuna importance
    'scale_pos_weight':    243.95,
    'bagging_temperature': 0.895,   # second most — 0.25 Optuna importance
    'random_strength':     1.995,
    'border_count':        240,
}
```

---

## 🌐 API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server + database status |
| POST | `/predict` | Single transaction prediction |
| POST | `/predict-batch` | Batch transaction predictions |
| GET | `/predictions/history` | Last 50 logged predictions |

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

### Risk Levels

| Risk Level | Threshold | What It Means |
|------------|-----------|---------------|
| 🟢 Low Risk | < 0.830 | Looks legitimate — carry on |
| 🟡 Elevated Risk | 0.830 – 0.9 | Flagged — model is on the fence |
| 🔴 High Risk | 0.9 – 0.99 | High confidence fraud |
| 🚨 Critical Risk | ≥ 0.99 | Near certain fraud — 84% of all fraud lands here |

---

## 🗂️ Project Structure

```
fraud-detection-catboost/
│
├── .github/
│   └── workflows/
│       └── tests.yml               # CI/CD — auto runs on every push
│
├── api/
│   ├── config.py                   # threshold, model paths, cat cols
│   ├── database.py                 # SQLAlchemy + Neon PostgreSQL
│   ├── main.py                     # FastAPI endpoints
│   ├── model_services.py           # inference + risk logic
│   └── schemas.py                  # Pydantic request/response models
│
├── images/
│   ├── cost_threshold.png
│   ├── cost_sensitivity_research.png
│   ├── feature_importance.png
│   ├── probability_distribution.png
│   └── pr_curve.png
│
├── models/                         # gitignored — download separately
│   ├── cb_model.cbm                # trained CatBoost model
│   └── preprocess_pipe.joblib      # sklearn preprocessing pipeline
│
├── notebooks/
│   └── fraud_detection.ipynb       # full pipeline — EDA → training → evaluation
│
├── src/
│   └── preprocess.py               # 40+ feature engineering transformations
│
├── tests/
│   └── test_api.py                 # 17 pytest tests
│
├── config.json                     # best Optuna hyperparameters
├── requirements.txt
└── README.md
```

---

## 💻 Quick Start

### Prerequisites
- Python 3.10+
- Git
- A [Neon](https://neon.tech) free account (PostgreSQL)
- Model files (`cb_model.cbm`, `preprocess_pipe.joblib`) — see step 4

### 1. Clone the repository
```bash
git clone https://github.com/mallapuabhiraj/fraud-detection-catboost.git
cd fraud-detection-catboost
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```
You should see `(venv)` in your terminal. If you don't — stop and fix this before continuing.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add model files
The `models/` folder is gitignored (files are too large for GitHub). Create it and add the model files manually:

```bash
# Windows
mkdir models

# Mac/Linux
mkdir -p models
```

Then copy `cb_model.cbm` and `preprocess_pipe.joblib` into the `models/` folder.

### 5. Set up environment variables
Create a `.env` file in the project root:

**Windows (PowerShell):**
```powershell
New-Item .env -ItemType File
Add-Content .env "DATABASE_URL=your_postgresql_connection_string"
```

**Mac/Linux:**
```bash
echo "DATABASE_URL=your_postgresql_connection_string" > .env
```

Replace `your_postgresql_connection_string` with your Neon connection string.

### 6. Run the API
```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger UI. If you see the docs page — you're done. 🎉

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

17 passed

---

## 📦 Dataset

[Kaggle Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) — Sparkov simulation.

| Split | Rows | Fraud Rate | Used For |
|-------|------|------------|----------|
| fraudTrain.csv | ~1.2M | ~0.6% | Training (base model) |
| fraudTest.csv | ~555K | ~0.6% | Evaluation (base model) |
| New Sparkov (Jan–Jun 2023) | ~185K | ~2.2% | Out-of-time drift test |

> Dataset not included in this repository — download from Kaggle.

---

## 🛠️ Skills Demonstrated

`CatBoost` `FastAPI` `PostgreSQL` `SQLAlchemy` `Pytest` `CI/CD` `GitHub Actions`
`Optuna` `Feature Engineering` `Imbalanced Learning` `Time-Series Validation`
`Cost-Sensitive ML` `Hyperparameter Tuning` `REST API Design` `Distribution Drift`
`Python` `Scikit-learn`
