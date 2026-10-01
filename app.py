import altair as alt
import pandas as pd
import streamlit as st

from propcompare.data import FURNISHING, LOCATIONS, PROPERTY_TYPES, clean, generate_dataset
from propcompare.model import predict, train_all
from propcompare.utils import fmt, per_sqft

st.set_page_config(page_title="Mumbai Property Compare", page_icon="🏙️", layout="wide")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return clean(generate_dataset())


@st.cache_resource(show_spinner="Training models…")
def load_models():
    return train_all(load_data())


df = load_data()
models = load_models()
LOCS = sorted(LOCATIONS)

st.title("🏙️ Mumbai & Navi Mumbai Property Compare")
st.caption("Estimate sale / rent prices and compare two properties side by side. "
           "Trained on a synthetic dataset modelled on Mumbai micro-markets — "
           "indicative only, not financial advice.")

listing = st.radio("I want to look at", ["Sale", "Rent"], horizontal=True)


def property_form(key: str, defaults: dict) -> dict:
    """Render inputs for one property and return a feature dict."""
    c1, c2 = st.columns(2)
    loc = c1.selectbox("Location", LOCS, index=LOCS.index(defaults["location"]), key=f"{key}_loc")
    ptype = c2.selectbox("Property type", PROPERTY_TYPES,
                         index=PROPERTY_TYPES.index(defaults["property_type"]), key=f"{key}_pt")
    c1, c2, c3 = st.columns(3)
    bhk = c1.number_input("BHK", 1, 6, defaults["bhk"], key=f"{key}_bhk")
    bath = c2.number_input("Bathrooms", 1, 8, defaults["bathrooms"], key=f"{key}_bath")
    area = c3.number_input("Area (sq ft)", 250, 6000, defaults["area_sqft"], step=25, key=f"{key}_area")
    c1, c2, c3 = st.columns(3)
    total = c1.number_input("Total floors", 1, 60, defaults["total_floors"], key=f"{key}_tf")
    floor = c2.number_input("Floor", 0, int(total), min(defaults["floor"], int(total)), key=f"{key}_fl")
    age = c3.number_input("Age (years)", 0.0, 50.0, float(defaults["age_years"]), step=0.5, key=f"{key}_age")
    c1, c2, c3 = st.columns(3)
    furn = c1.selectbox("Furnishing", FURNISHING, index=defaults["furn_idx"], key=f"{key}_fu")
    parking = c2.selectbox("Parking spots", [0, 1, 2], index=defaults["parking"], key=f"{key}_pk")
    metro = c3.number_input("Metro distance (km)", 0.0, 15.0, float(defaults["metro_dist_km"]),
                            step=0.1, key=f"{key}_mt")
    school = st.slider("School distance (km)", 0.3, 5.0, float(defaults["school_dist_km"]), 0.1,
                       key=f"{key}_sc")
    a1, a2, a3, a4 = st.columns(4)
    lift = int(a1.checkbox("Lift", defaults["lift"], key=f"{key}_lift"))
    gym = int(a2.checkbox("Gym", defaults["gym"], key=f"{key}_gym"))
    pool = int(a3.checkbox("Pool", defaults["pool"], key=f"{key}_pool"))
    sec = int(a4.checkbox("24x7 security", defaults["security"], key=f"{key}_sec"))
    return dict(location=loc, city=LOCATIONS[loc][0], property_type=ptype, bhk=int(bhk),
                bathrooms=int(bath), area_sqft=float(area), floor=int(floor),
                total_floors=int(total), age_years=float(age), furnishing=furn,
                parking=int(parking), lift=lift, gym=gym, pool=pool, security=sec,
                metro_dist_km=float(metro), school_dist_km=float(school))


A_DEF = dict(location="Andheri West", property_type="Apartment", bhk=2, bathrooms=2,
             area_sqft=850, floor=8, total_floors=20, age_years=4.0, furn_idx=1, parking=1,
             metro_dist_km=0.8, school_dist_km=1.0, lift=True, gym=True, pool=False, security=True)
B_DEF = dict(A_DEF, location="Kharghar", area_sqft=950, floor=10, total_floors=22,
             age_years=2.0, metro_dist_km=1.5, pool=True)

tab_pred, tab_cmp, tab_mkt, tab_model = st.tabs(
    ["💰 Price estimate", "⚖️ Compare two", "📊 Market explorer", "🧠 Model"])

# ---------------------------------------------------------------- estimate
with tab_pred:
    left, right = st.columns([3, 2], gap="large")
    with left:
        prop = property_form("single", A_DEF)
    r = predict(models, listing, prop)
    with right:
        st.subheader("Estimated " + ("price" if listing == "Sale" else "monthly rent"))
        st.metric("Estimate", fmt(listing, r["estimate"]))
        st.write(f"**80% range:** {fmt(listing, r['low'])} – {fmt(listing, r['high'])}")
        unit = "per sq ft" if listing == "Sale" else "per sq ft / month"
        st.write(f"**≈ ₹{per_sqft(listing, r['estimate'], prop['area_sqft']):,.0f}** {unit}")
        loc_sub = df[(df.location == prop["location"]) & (df.listing_type == listing)]
        if len(loc_sub):
            med = loc_sub["price"].median()
            st.write(f"Median {listing.lower()} listing in {prop['location']}: **{fmt(listing, med)}**")

# ---------------------------------------------------------------- compare
with tab_cmp:
    ca, cb = st.columns(2, gap="large")
    with ca:
        st.subheader("Property A")
        pa = property_form("A", A_DEF)
    with cb:
        st.subheader("Property B")
        pb = property_form("B", B_DEF)

    ra, rb = predict(models, listing, pa), predict(models, listing, pb)
    pps_a = per_sqft(listing, ra["estimate"], pa["area_sqft"])
    pps_b = per_sqft(listing, rb["estimate"], pb["area_sqft"])

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("A — " + pa["location"], fmt(listing, ra["estimate"]))
    m2.metric("B — " + pb["location"], fmt(listing, rb["estimate"]),
              delta=f"{(rb['estimate'] / ra['estimate'] - 1) * 100:+.1f}% vs A", delta_color="off")
    cheaper = "A" if pps_a < pps_b else "B"
    m3.metric("Better value per sq ft", f"Property {cheaper}",
              delta=f"₹{abs(pps_a - pps_b):,.0f} lower per sq ft", delta_color="off")

    table = pd.DataFrame({
        "": ["Location", "Type", "BHK / Bath", "Area (sq ft)", "Floor", "Age (yrs)",
             "Furnishing", "Metro (km)", "Estimate", "80% range", "₹ per sq ft"],
        "Property A": [pa["location"], pa["property_type"], f"{pa['bhk']} / {pa['bathrooms']}",
                       f"{pa['area_sqft']:,.0f}", f"{pa['floor']}/{pa['total_floors']}",
                       pa["age_years"], pa["furnishing"], pa["metro_dist_km"],
                       fmt(listing, ra["estimate"]),
                       f"{fmt(listing, ra['low'])} – {fmt(listing, ra['high'])}", f"{pps_a:,.0f}"],
        "Property B": [pb["location"], pb["property_type"], f"{pb['bhk']} / {pb['bathrooms']}",
                       f"{pb['area_sqft']:,.0f}", f"{pb['floor']}/{pb['total_floors']}",
                       pb["age_years"], pb["furnishing"], pb["metro_dist_km"],
                       fmt(listing, rb["estimate"]),
                       f"{fmt(listing, rb['low'])} – {fmt(listing, rb['high'])}", f"{pps_b:,.0f}"],
    }).astype(str)
    st.dataframe(table, hide_index=True, width="stretch")

    chart_df = pd.DataFrame({"Property": ["A", "B"], "₹ per sq ft": [pps_a, pps_b]})
    st.altair_chart(
        alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Property:N", axis=alt.Axis(labelAngle=0)), y="₹ per sq ft:Q",
            color=alt.Color("Property:N", legend=None), tooltip=["Property", "₹ per sq ft"]
        ).properties(height=250, title="Price per sq ft"),
        width="stretch")

# ---------------------------------------------------------------- market
with tab_mkt:
    sub = df[df.listing_type == listing]
    st.subheader(f"Median {listing.lower()} price by location")
    by_loc = sub.groupby(["location", "city"], as_index=False)["price"].median()
    by_loc["label"] = [fmt(listing, v) for v in by_loc["price"]]
    st.altair_chart(
        alt.Chart(by_loc).mark_bar().encode(
            y=alt.Y("location:N", sort="-x", title=None), x=alt.X("price:Q", title="Median price"),
            color=alt.Color("city:N", title="City"), tooltip=["location", "city", "label"]
        ).properties(height=520), width="stretch")

    c1, c2 = st.columns(2)
    c1.subheader("Area vs price")
    c1.altair_chart(
        alt.Chart(sub).mark_circle(opacity=0.5, size=35).encode(
            x="area_sqft:Q", y="price:Q", color=alt.Color("bhk:O", title="BHK"),
            tooltip=["location", "bhk", "area_sqft", "price"]).interactive(),
        width="stretch")
    c2.subheader("Price by BHK")
    c2.altair_chart(
        alt.Chart(sub).mark_boxplot().encode(x=alt.X("bhk:O", title="BHK"), y="price:Q"),
        width="stretch")

# ---------------------------------------------------------------- model
with tab_model:
    st.write("Two gradient-boosting regressors (Sale and Rent) trained on log-price, with "
             "engineered features (area per BHK, floor ratio, amenity score, location tier, "
             "leakage-free location target-encoding).")
    for lt in ("Sale", "Rent"):
        b = models[lt]
        st.subheader(f"{lt} model")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² (test)", f"{b.metrics['R2']:.3f}")
        c2.metric("MAPE", f"{b.metrics['MAPE_%']:.1f}%")
        c3.metric("MAE", fmt(lt, b.metrics["MAE"]))
        c4.metric("Train / test rows", f"{b.metrics['train_rows']} / {b.metrics['test_rows']}")
        imp = b.importances.head(10).reset_index()
        imp.columns = ["feature", "importance"]
        st.altair_chart(
            alt.Chart(imp).mark_bar().encode(
                y=alt.Y("feature:N", sort="-x", title=None), x="importance:Q"
            ).properties(height=260), width="stretch")
    with st.expander("Preview dataset"):
        st.dataframe(df.head(100), width="stretch")
