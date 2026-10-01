# 🏙️ Mumbai & Navi Mumbai Property Compare

A Streamlit app that estimates **sale prices and monthly rents** for properties in
Mumbai and Navi Mumbai and lets you **compare two properties side by side**.

**Live app:** _add your Streamlit URL here after deploying_

## Features
- **Price estimate** – point estimate, 80% range and ₹/sq ft for any property
- **Compare two** – two properties, price difference, value per sq ft
- **Market explorer** – median prices by location, area vs price, BHK distribution
- **Model tab** – test-set R², MAPE, MAE and feature importances

## How it works
| Step | Detail |
|---|---|
| Data | Synthetic dataset of 2,500 listings across 25 localities (`propcompare/data.py`). Swap in real scraped data by replacing `generate_dataset()` with `pd.read_csv(...)` |
| Cleaning | De-duplication, area outliers, price-per-sq-ft outliers by locality, median/mode imputation, logical checks |
| Features | Area per BHK, floor ratio, amenity score, location tier, leakage-free location target-encoding, property-type dummies |
| Model | Separate `GradientBoostingRegressor` for Sale and Rent on log-price (`propcompare/model.py`) |
| Results | Sale R² ≈ 0.98 · MAPE ≈ 8%  /  Rent R² ≈ 0.98 · MAPE ≈ 7% (held-out 20%) |

> ⚠️ The data is synthetic and the estimates are **indicative only** – not valuation or financial advice.

## Run locally
```bash
git clone https://github.com/<your-username>/mumbai-property-compare.git
cd mumbai-property-compare
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
```
app.py                  Streamlit UI
propcompare/
  data.py               dataset generation + cleaning
  model.py              feature engineering, training, prediction
  utils.py              ₹ formatting helpers
data/                   raw + cleaned CSVs
notebooks/              original notebook export (EDA)
requirements.txt
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io> → **Create app** → pick the repo, branch `main`, main file `app.py`.
3. Click **Deploy**.

## Notes on changes from the original notebook
- Fixed a unit bug in the price generator (sale prices were ~100× too low).
- Location target-encoding is now fitted on the training split only (the notebook computed it on all data and mixed Sale and Rent prices).
- Outlier removal now works on price per sq ft by locality, so the injected bad prices are actually caught.
