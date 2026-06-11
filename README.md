# Netflix Prize — Recommendation System

A personalized movie recommendation engine built on the Netflix Prize Dataset.
Developed as part of Cultural Council Open Projects 2026, IIT Roorkee.

---

## What This Does

This project builds two recommendation models:
1. **User-Based Collaborative Filtering** — finds similar users and recommends what they liked
2. **SVD (Matrix Factorization)** — learns latent user and movie representations

Both models are evaluated on **RMSE** (rating accuracy) and **MAP@10** (ranking quality).

---

## Project Structure

```
netflix-recommender/
├── data/               ← put dataset files here (see data/README.md)
├── src/
│   ├── config.py       ← global settings (sample size, thresholds, etc.)
│   ├── data_loader.py  ← parses Netflix Prize file format
│   ├── preprocessing.py
│   ├── eda.py          ← 5 EDA plots
│   ├── evaluation.py   ← RMSE, MAE, MAP@10
│   ├── recommender.py  ← generate + display recommendations
│   └── models/
│       ├── user_cf.py    ← User-Based CF (from scratch)
│       └── svd_model.py  ← SVD via Surprise library
├── outputs/figures/    ← auto-generated plots
├── report/             ← technical report (PDF-ready markdown)
├── presentation/       ← 8-slide deck (markdown)
├── main.py             ← run everything
└── requirements.txt
```

---

## Setup

### 1. Clone / download this repo

```bash
git clone <repo-url>
cd netflix-recommender
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

See `data/README.md` for instructions.  
Short version: download from https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
and place the extracted files in the `data/` folder.

---

## Running the Pipeline

```bash
python main.py
```

This will:
- Load and sample 500K ratings from the dataset
- Run full EDA (saves 5 plots to `outputs/figures/`)
- Train User-Based CF and SVD models
- Evaluate RMSE, MAE, and MAP@10
- Generate and display Top-10 recommendations for sample users

Expected runtime: ~5–15 minutes depending on your machine.

---

## Key Results

| Model | RMSE | MAE | MAP@10 |
|-------|------|-----|--------|
| Global Mean Baseline | 1.12 | 0.93 | ~0.05 |
| User-Based CF (k=30) | 0.96 | 0.75 | — |
| SVD (100 factors) | 0.88 | 0.69 | 0.18 |

SVD outperforms User-CF on rating prediction accuracy and achieves a MAP@10 of 0.18
(vs. 0.05 for random recommendations — a 3.6× improvement).

---

## Adjusting Parameters

Edit `src/config.py`:

```python
SAMPLE_SIZE = 500_000      # increase for better results (but slower)
TOP_K = 10                 # number of recommendations per user
RELEVANCE_THRESHOLD = 3.5  # what counts as a "liked" movie for MAP@10
```

Edit model hyperparameters directly in `main.py` when instantiating the models.

---

## Deliverables

| Deliverable | Location |
|-------------|----------|
| Technical Report (PDF) | `report/technical_report.md` (export to PDF) |
| Presentation (PDF) | `presentation/slides.md` (export to PDF) |
| Source Code | This repository |
| EDA Figures | `outputs/figures/` |

---

## Notes

- The full Netflix Prize dataset has 100M+ ratings. We use 500K by default.
  You can increase `SAMPLE_SIZE` in `config.py` if your machine can handle it.
- SVD training on 500K ratings takes about 3–5 minutes on a standard laptop.
- User-Based CF prediction is slower (recommend using test samples of ≤ 5K rows).

---

## References

- Koren, Y. (2009). Collaborative Filtering with Temporal Dynamics. KDD '09.
- Funk, S. (2006). Netflix Update: Try This at Home.
- Surprise library: https://surpriselib.com/
- Netflix Prize Dataset: https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
