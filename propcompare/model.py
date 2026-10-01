"""Feature engineering + training. Separate Sale and Rent models (log-price target).

Location target-encoding is fit on the training split only (no leakage).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

from .data import FURNISHING, LOCATIONS, PROPERTY_TYPES

RANDOM_STATE = 42
PREMIUM = ["Worli", "Bandra West", "Lower Parel", "Juhu", "Dadar"]
MID = ["Andheri West", "Powai", "Goregaon West", "Santacruz West", "Malad West",
       "Borivali West", "Vashi", "Nerul", "Belapur"]
FURN_MAP = {k: i for i, k in enumerate(FURNISHING)}


def _tier(loc: str) -> int:
    if loc in PREMIUM:
        return 3
    if loc in MID:
        return 2
    return 1 if LOCATIONS[loc][0] == "Mumbai" else 0


def engineer(df: pd.DataFrame, loc_enc: dict) -> pd.DataFrame:
    """Raw listing columns -> model features. `loc_enc` maps location -> log(price/sqft)."""
    f = pd.DataFrame(index=df.index)
    for c in ["area_sqft", "bhk", "bathrooms", "floor", "total_floors", "age_years",
              "parking", "lift", "gym", "pool", "security", "metro_dist_km", "school_dist_km"]:
        f[c] = df[c].astype(float)
    f["area_per_bhk"] = f["area_sqft"] / f["bhk"]
    f["floor_ratio"] = f["floor"] / (f["total_floors"] + 1)
    f["is_top_floor"] = (f["floor"] == f["total_floors"]).astype(int)
    f["is_high_rise"] = (f["total_floors"] >= 20).astype(int)
    f["furnishing_ord"] = df["furnishing"].map(FURN_MAP)
    f["amenity_score"] = (f["lift"] + 1.5 * f["gym"] + 2 * f["pool"]
                          + 0.5 * f["security"] + f["parking"])
    f["location_tier"] = df["location"].map(_tier)
    f["is_navi_mumbai"] = (df["location"].map(lambda l: LOCATIONS[l][0]) == "Navi Mumbai").astype(int)
    f["location_enc"] = df["location"].map(loc_enc)
    for p in PROPERTY_TYPES:
        f[f"ptype_{p}"] = (df["property_type"] == p).astype(int)
    return f


@dataclass
class Bundle:
    model: GradientBoostingRegressor
    loc_enc: dict
    features: list
    metrics: dict
    sigma: float                      # std of log-residuals on the test set
    importances: pd.Series = field(default=None)


def train_one(d: pd.DataFrame) -> Bundle:
    train, test = train_test_split(d, test_size=0.2, random_state=RANDOM_STATE)
    enc = np.log(train["price"] / train["area_sqft"]).groupby(train["location"]).mean()
    default = float(enc.mean())
    loc_enc = {l: float(enc.get(l, default)) for l in LOCATIONS}

    Xtr, Xte = engineer(train, loc_enc), engineer(test, loc_enc)
    ytr, yte = np.log1p(train["price"]), np.log1p(test["price"])

    model = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                                      subsample=0.8, random_state=RANDOM_STATE)
    model.fit(Xtr, ytr)
    pred = np.expm1(model.predict(Xte))
    metrics = {
        "R2": float(r2_score(test["price"], pred)),
        "MAE": float(mean_absolute_error(test["price"], pred)),
        "MAPE_%": float(mean_absolute_percentage_error(test["price"], pred) * 100),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
    }
    sigma = float(np.std(yte - model.predict(Xte)))
    imp = pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    return Bundle(model, loc_enc, list(Xtr.columns), metrics, sigma, imp)


def train_all(clean_df: pd.DataFrame) -> dict:
    return {lt: train_one(clean_df[clean_df["listing_type"] == lt]) for lt in ("Sale", "Rent")}


def predict(bundles: dict, listing_type: str, prop: dict) -> dict:
    """Point estimate + ~80% interval. Sale in Rs Lakhs, Rent in Rs/month."""
    b = bundles[listing_type]
    row = pd.DataFrame([prop])
    X = engineer(row, b.loc_enc)[b.features]
    mu = float(b.model.predict(X)[0])
    z = 1.2816  # 80% interval
    return {"estimate": float(np.expm1(mu)),
            "low": float(np.expm1(mu - z * b.sigma)),
            "high": float(np.expm1(mu + z * b.sigma))}
