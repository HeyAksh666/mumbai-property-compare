"""Synthetic Mumbai / Navi Mumbai listing generator + cleaning pipeline.

Logic taken from the original notebook (property_comparehelp). Replace
`generate_dataset()` with `pd.read_csv(...)` when real scraped data is available.
"""
import numpy as np
import pandas as pd

# location -> (city, base sale rate in Rs '000 per sq ft, base monthly rent in Rs '000 for a 600 sq ft flat)
LOCATIONS = {
    "Bandra West": ("Mumbai", 25.0, 75), "Juhu": ("Mumbai", 22.0, 65),
    "Andheri West": ("Mumbai", 18.0, 55), "Andheri East": ("Mumbai", 15.0, 45),
    "Powai": ("Mumbai", 16.0, 48), "Malad West": ("Mumbai", 13.0, 38),
    "Borivali West": ("Mumbai", 12.0, 35), "Goregaon West": ("Mumbai", 14.0, 42),
    "Thane West": ("Mumbai", 11.0, 32), "Kurla": ("Mumbai", 9.0, 28),
    "Dadar": ("Mumbai", 20.0, 60), "Worli": ("Mumbai", 28.0, 85),
    "Lower Parel": ("Mumbai", 26.0, 80), "Santacruz West": ("Mumbai", 21.0, 62),
    "Chembur": ("Mumbai", 12.5, 37), "Vashi": ("Navi Mumbai", 10.5, 30),
    "Kharghar": ("Navi Mumbai", 8.5, 25), "Panvel": ("Navi Mumbai", 6.5, 18),
    "Nerul": ("Navi Mumbai", 9.5, 27), "Belapur": ("Navi Mumbai", 9.0, 26),
    "Airoli": ("Navi Mumbai", 8.0, 23), "Ghansoli": ("Navi Mumbai", 7.5, 22),
    "Kopar Khairane": ("Navi Mumbai", 8.2, 24), "Ulwe": ("Navi Mumbai", 6.0, 17),
    "Dronagiri": ("Navi Mumbai", 5.5, 15),
}
PROPERTY_TYPES = ["Apartment", "Studio", "Villa", "Row House", "Penthouse"]
FURNISHING = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]


def _make_row(loc: str) -> dict:
    city, base_sale, base_rent = LOCATIONS[loc]
    prop_type = np.random.choice(PROPERTY_TYPES, p=[0.60, 0.10, 0.12, 0.10, 0.08])
    listing = np.random.choice(["Sale", "Rent"], p=[0.55, 0.45])

    if prop_type == "Studio":
        bhk = 1
    elif prop_type in ("Villa", "Row House"):
        bhk = np.random.choice([3, 4, 5], p=[0.4, 0.4, 0.2])
    elif prop_type == "Penthouse":
        bhk = np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
    else:
        bhk = np.random.choice([1, 2, 3, 4], p=[0.20, 0.40, 0.30, 0.10])

    bathrooms = min(bhk + np.random.randint(0, 2), 6)
    area = round(bhk * np.random.uniform(380, 620) + np.random.normal(0, 50), 1)
    area = max(area, 250)
    floor = np.random.randint(0, 35)
    tot_floors = floor + np.random.randint(1, 15)
    age = round(np.random.exponential(7), 1)
    furnishing = np.random.choice(FURNISHING, p=[0.30, 0.40, 0.30])
    parking = np.random.choice([0, 1, 2], p=[0.25, 0.55, 0.20])
    lift = 1 if tot_floors > 4 else int(np.random.rand() > 0.3)
    gym = int(np.random.rand() > 0.55)
    pool = int(np.random.rand() > 0.75)
    security = int(np.random.rand() > 0.40)
    metro_dist = round(np.random.exponential(1.2), 2)
    school_dist = round(np.random.uniform(0.3, 3.5), 2)

    area_price = base_sale * 1000 * area          # Rs
    bhk_mult = 1 + (bhk - 2) * 0.12
    floor_bonus = 1 + floor * 0.003
    age_disc = max(0.70, 1 - age * 0.015)
    furnish_m = {"Unfurnished": 1.0, "Semi-Furnished": 1.05, "Fully Furnished": 1.12}[furnishing]
    amenity_m = 1 + 0.02 * gym + 0.03 * pool + 0.01 * security + 0.02 * lift
    metro_disc = max(0.88, 1 - metro_dist * 0.04)
    type_m = {"Apartment": 1.0, "Studio": 0.85, "Villa": 1.35,
              "Row House": 1.20, "Penthouse": 1.50}[prop_type]
    noise = np.random.normal(1.0, 0.06)

    if listing == "Sale":
        price = (area_price * bhk_mult * floor_bonus * age_disc * furnish_m
                 * amenity_m * metro_disc * type_m * noise)
        price = round(price / 100000, 2)            # Lakhs
    else:
        price = (base_rent * area / 600 * bhk_mult * furnish_m * amenity_m
                 * metro_disc * type_m * noise * 1000)
        price = round(price / 1000) * 1000          # Rs / month

    return {
        "location": loc, "city": city, "property_type": prop_type, "bhk": bhk,
        "bathrooms": bathrooms, "area_sqft": area, "floor": floor,
        "total_floors": tot_floors, "age_years": age, "furnishing": furnishing,
        "parking": parking, "lift": lift, "gym": gym, "pool": pool,
        "security": security, "metro_dist_km": metro_dist,
        "school_dist_km": school_dist, "listing_type": listing, "price": price,
    }


def generate_dataset(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    names = list(LOCATIONS)
    probs = np.array([3, 2, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2,
                      4, 4, 3, 3, 3, 2, 2, 2, 2, 1], dtype=float)
    probs /= probs.sum()
    locs = np.random.choice(names, size=n, p=probs)
    df = pd.DataFrame([_make_row(l) for l in locs])

    # realistic missing values + data-entry outliers (as in the notebook)
    for col, rate in [("age_years", 0.05), ("parking", 0.03),
                      ("metro_dist_km", 0.04), ("floor", 0.02)]:
        df.loc[df.sample(frac=rate, random_state=42).index, col] = np.nan
    df.loc[df.sample(10, random_state=7).index, "area_sqft"] *= 10
    df.loc[df.sample(5, random_state=8).index, "price"] *= 0.01
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df.drop_duplicates().copy()

    # area outliers
    d = d[d["area_sqft"] <= d["area_sqft"].quantile(0.99)]

    # price outliers: compare price-per-sqft with the location+listing median
    d["_pps"] = d["price"] / d["area_sqft"]
    med = d.groupby(["location", "listing_type"])["_pps"].transform("median")
    d = d[(d["_pps"] > 0.4 * med) & (d["_pps"] < 2.5 * med)].drop(columns="_pps")

    # impute
    for c in d.select_dtypes(include=np.number).columns:
        d[c] = d[c].fillna(d[c].median())
    for c in d.select_dtypes(include="object").columns:
        d[c] = d[c].fillna(d[c].mode()[0])

    # logical consistency
    d = d[(d["bathrooms"] <= d["bhk"] + 2) & (d["floor"] <= d["total_floors"])
          & (d["area_sqft"] >= 150)]
    return d.reset_index(drop=True)
