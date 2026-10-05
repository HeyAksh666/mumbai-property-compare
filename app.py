import altair as alt
import pandas as pd
import streamlit as st

from propcompare.data import FURNISHING, LOCATIONS, PROPERTY_TYPES, clean, generate_dataset
from propcompare.model import predict, train_all
from propcompare.utils import fmt, per_sqft


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="PropCompare — Mumbai Real Estate Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PREMIUM UI
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --bg: #07111f;
    --surface: rgba(16, 28, 47, .78);
    --surface-2: rgba(22, 37, 61, .72);
    --border: rgba(255,255,255,.09);
    --text: #f4f7fb;
    --muted: #9eabc0;
    --accent: #5eead4;
    --accent-2: #60a5fa;
    --gold: #f4c95d;
}

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 5%, rgba(96,165,250,.14), transparent 30%),
        radial-gradient(circle at 90% 12%, rgba(94,234,212,.10), transparent 25%),
        linear-gradient(135deg, #050b14 0%, #07111f 55%, #0a1424 100%);
    color: var(--text);
}

.block-container {
    max-width: 1450px;
    padding: 2.2rem 3rem 5rem;
}

/* subtle animated background */
.stApp:before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.018) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: linear-gradient(to bottom, black, transparent 75%);
    animation: gridMove 18s linear infinite;
}

@keyframes gridMove {
    from { transform: translateY(0); }
    to { transform: translateY(44px); }
}

/* Hero */
.hero {
    position: relative;
    padding: 2.2rem 2.3rem;
    border: 1px solid var(--border);
    border-radius: 28px;
    overflow: hidden;
    background:
        radial-gradient(circle at 82% 20%, rgba(94,234,212,.14), transparent 24%),
        radial-gradient(circle at 12% 80%, rgba(96,165,250,.14), transparent 28%),
        rgba(9,19,34,.84);
    box-shadow: 0 25px 80px rgba(0,0,0,.30);
    transform: translateZ(0);
}

.hero:after {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    right: -80px;
    top: -100px;
    border-radius: 50%;
    border: 1px solid rgba(94,234,212,.22);
    box-shadow:
        0 0 0 30px rgba(94,234,212,.025),
        0 0 0 60px rgba(94,234,212,.018);
    animation: pulse 5s ease-in-out infinite;
}

@keyframes pulse {
    50% { transform: scale(1.12); opacity: .65; }
}

.eyebrow {
    color: var(--accent);
    font-size: .76rem;
    font-weight: 800;
    letter-spacing: .18em;
    text-transform: uppercase;
}

.hero h1 {
    font-family: "Manrope", sans-serif;
    font-size: clamp(2rem, 4vw, 4rem);
    line-height: 1.02;
    margin: .5rem 0 .8rem;
    letter-spacing: -.045em;
}

.hero p {
    max-width: 780px;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.7;
}

.badge {
    display: inline-block;
    margin-top: 1rem;
    padding: .42rem .75rem;
    border: 1px solid rgba(244,201,93,.25);
    border-radius: 999px;
    color: #f8dda0;
    background: rgba(244,201,93,.06);
    font-size: .78rem;
}

/* Section headings */
.section-label {
    color: var(--muted);
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin: 1.8rem 0 .45rem;
}

.section-title {
    font-family: "Manrope", sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -.025em;
}

/* Cards */
.card {
    background: linear-gradient(145deg, rgba(20,34,56,.86), rgba(10,21,37,.82));
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1.25rem;
    box-shadow: 0 14px 45px rgba(0,0,0,.18);
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}

.card:hover {
    transform: translateY(-4px);
    border-color: rgba(94,234,212,.24);
    box-shadow: 0 20px 60px rgba(0,0,0,.28);
}

.kpi {
    min-height: 120px;
}

.kpi-label {
    color: var(--muted);
    font-size: .78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.kpi-value {
    margin-top: .35rem;
    font-family: "Manrope", sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -.04em;
}

.kpi-sub {
    color: #7dd3fc;
    font-size: .78rem;
    margin-top: .3rem;
}

/* Streamlit widgets */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] > div {
    background: rgba(7,17,31,.76) !important;
    border-color: var(--border) !important;
    border-radius: 12px !important;
}

.stRadio > div {
    gap: .4rem;
}

.stRadio label {
    border: 1px solid var(--border);
    background: rgba(10,20,35,.65);
    border-radius: 999px;
    padding: .35rem .75rem;
}

button[kind="secondary"] {
    border-radius: 12px;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 700;
    color: var(--muted);
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent);
}

/* Tables */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
}

/* Dividers */
hr {
    border-color: var(--border);
}

/* Fade-in sections */
.fade {
    animation: fadeUp .55s ease both;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Footer */
.footer {
    text-align: center;
    color: #68758a;
    font-size: .75rem;
    padding: 2.5rem 0 1rem;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background: transparent !important;}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA / MODELS
# ============================================================
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return clean(generate_dataset())


@st.cache_resource(show_spinner="Training pricing models…")
def load_models():
    return train_all(load_data())


df = load_data()
models = load_models()
LOCS = sorted(LOCATIONS)


# ============================================================
# HELPERS
# ============================================================
def money_card(title, value, subtitle="", icon="₹"):
    st.markdown(
        f"""
        <div class="card kpi fade">
            <div class="kpi-label">{icon} &nbsp; {title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def property_form(key: str, defaults: dict) -> dict:
    """Render a clean property input panel and return model features."""

    st.markdown('<div class="section-label">Property profile</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    loc = c1.selectbox(
        "Location",
        LOCS,
        index=LOCS.index(defaults["location"]),
        key=f"{key}_loc",
    )

    ptype = c2.selectbox(
        "Property type",
        PROPERTY_TYPES,
        index=PROPERTY_TYPES.index(defaults["property_type"]),
        key=f"{key}_pt",
    )

    c1, c2, c3 = st.columns(3)

    bhk = c1.number_input(
        "BHK", 1, 6, defaults["bhk"], key=f"{key}_bhk"
    )

    bath = c2.number_input(
        "Bathrooms", 1, 8, defaults["bathrooms"], key=f"{key}_bath"
    )

    area = c3.number_input(
        "Area (sq ft)",
        250,
        6000,
        defaults["area_sqft"],
        step=25,
        key=f"{key}_area",
    )

    c1, c2, c3 = st.columns(3)

    total = c1.number_input(
        "Total floors", 1, 60, defaults["total_floors"], key=f"{key}_tf"
    )

    floor = c2.number_input(
        "Floor",
        0,
        int(total),
        min(defaults["floor"], int(total)),
        key=f"{key}_fl",
    )

    age = c3.number_input(
        "Property age",
        0.0,
        50.0,
        float(defaults["age_years"]),
        step=0.5,
        key=f"{key}_age",
    )

    c1, c2, c3 = st.columns(3)

    furn = c1.selectbox(
        "Furnishing",
        FURNISHING,
        index=defaults["furn_idx"],
        key=f"{key}_fu",
    )

    parking = c2.selectbox(
        "Parking spots",
        [0, 1, 2],
        index=defaults["parking"],
        key=f"{key}_pk",
    )

    metro = c3.number_input(
        "Metro distance (km)",
        0.0,
        15.0,
        float(defaults["metro_dist_km"]),
        step=0.1,
        key=f"{key}_mt",
    )

    school = st.slider(
        "School distance (km)",
        0.3,
        5.0,
        float(defaults["school_dist_km"]),
        0.1,
        key=f"{key}_sc",
    )

    st.markdown('<div class="section-label">Amenities</div>', unsafe_allow_html=True)

    a1, a2, a3, a4 = st.columns(4)

    lift = int(a1.checkbox("Lift", defaults["lift"], key=f"{key}_lift"))
    gym = int(a2.checkbox("Gym", defaults["gym"], key=f"{key}_gym"))
    pool = int(a3.checkbox("Pool", defaults["pool"], key=f"{key}_pool"))
    sec = int(a4.checkbox("24×7 Security", defaults["security"], key=f"{key}_sec"))

    return dict(
        location=loc,
        city=LOCATIONS[loc][0],
        property_type=ptype,
        bhk=int(bhk),
        bathrooms=int(bath),
        area_sqft=float(area),
        floor=int(floor),
        total_floors=int(total),
        age_years=float(age),
        furnishing=furn,
        parking=int(parking),
        lift=lift,
        gym=gym,
        pool=pool,
        security=sec,
        metro_dist_km=float(metro),
        school_dist_km=float(school),
    )


def comparison_score(a, b):
    """Simple value score based on predicted price per sq ft."""
    if b == 0:
        return 0
    return ((a - b) / a) * 100


# ============================================================
# DEFAULTS
# ============================================================
A_DEF = dict(
    location="Andheri West",
    property_type="Apartment",
    bhk=2,
    bathrooms=2,
    area_sqft=850,
    floor=8,
    total_floors=20,
    age_years=4.0,
    furn_idx=1,
    parking=1,
    metro_dist_km=0.8,
    school_dist_km=1.0,
    lift=True,
    gym=True,
    pool=False,
    security=True,
)

B_DEF = dict(
    A_DEF,
    location="Kharghar",
    area_sqft=950,
    floor=10,
    total_floors=22,
    age_years=2.0,
    metro_dist_km=1.5,
    pool=True,
)


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero fade">
        <div class="eyebrow">Property intelligence platform</div>
        <h1>Compare Mumbai property<br>with data, not guesswork.</h1>
        <p>
            Estimate sale and rental values, compare two properties,
            explore micro-markets and understand which features are
            influencing the model's prediction.
        </p>
        <span class="badge">MODEL DATASET • SIMULATED MARKET DATA • INDICATIVE ONLY</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Transaction mode</div>', unsafe_allow_html=True)

listing = st.radio(
    "Choose market",
    ["Sale", "Rent"],
    horizontal=True,
    label_visibility="collapsed",
)

tab_pred, tab_cmp, tab_mkt, tab_model = st.tabs(
    [
        "💰 Estimate",
        "⚖️ Compare",
        "📊 Market intelligence",
        "🧠 Model",
    ]
)


# ============================================================
# ESTIMATE
# ============================================================
with tab_pred:
    left, right = st.columns([1.45, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">Build your property</div>', unsafe_allow_html=True)
        prop = property_form("single", A_DEF)

    result = predict(models, listing, prop)

    with right:
        st.markdown('<div class="section-title">Valuation snapshot</div>', unsafe_allow_html=True)

        money_card(
            "Estimated " + ("sale price" if listing == "Sale" else "monthly rent"),
            fmt(listing, result["estimate"]),
            "Model prediction",
            "◈",
        )

        c1, c2 = st.columns(2)

        with c1:
            money_card(
                "Lower bound",
                fmt(listing, result["low"]),
                "Approx. 80% prediction range",
                "↓",
            )

        with c2:
            money_card(
                "Upper bound",
                fmt(listing, result["high"]),
                "Approx. 80% prediction range",
                "↑",
            )

        pps = per_sqft(listing, result["estimate"], prop["area_sqft"])

        money_card(
            "Price intensity",
            f"₹{pps:,.0f}",
            "per sq ft" + (" / month" if listing == "Rent" else ""),
            "⌁",
        )

        loc_sub = df[
            (df.location == prop["location"]) &
            (df.listing_type == listing)
        ]

        if len(loc_sub):
            med = loc_sub["price"].median()

            diff = ((result["estimate"] / med) - 1) * 100 if med else 0

            st.markdown(
                f"""
                <div class="card fade" style="margin-top:1rem;">
                    <div class="kpi-label">LOCAL MARKET CHECK</div>
                    <div style="font-size:1rem;margin-top:.45rem;">
                        Median in <b>{prop["location"]}</b>:
                        <b>{fmt(listing, med)}</b>
                    </div>
                    <div style="color:#7dd3fc;font-size:.82rem;margin-top:.35rem;">
                        Model vs local median: {diff:+.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "This is a model estimate, not a property appraisal, listing quote, "
            "investment recommendation, or guaranteed market price."
        )


# ============================================================
# COMPARE
# ============================================================
with tab_cmp:
    st.markdown('<div class="section-title">Two-property decision view</div>', unsafe_allow_html=True)
    st.caption("Change the inputs to see how location, size, amenities and property age affect the comparison.")

    ca, cb = st.columns(2, gap="large")

    with ca:
        st.markdown("### Property A")
        pa = property_form("A", A_DEF)

    with cb:
        st.markdown("### Property B")
        pb = property_form("B", B_DEF)

    ra = predict(models, listing, pa)
    rb = predict(models, listing, pb)

    pps_a = per_sqft(listing, ra["estimate"], pa["area_sqft"])
    pps_b = per_sqft(listing, rb["estimate"], pb["area_sqft"])

    st.divider()

    cheaper = "A" if pps_a < pps_b else "B"
    price_diff = abs(pps_a - pps_b)

    m1, m2, m3 = st.columns(3)

    with m1:
        money_card(
            f"A — {pa['location']}",
            fmt(listing, ra["estimate"]),
            f"₹{pps_a:,.0f} per sq ft",
            "A",
        )

    with m2:
        delta = ((rb["estimate"] / ra["estimate"]) - 1) * 100 if ra["estimate"] else 0
        money_card(
            f"B — {pb['location']}",
            fmt(listing, rb["estimate"]),
            f"{delta:+.1f}% vs Property A",
            "B",
        )

    with m3:
        money_card(
            "Better value",
            f"Property {cheaper}",
            f"₹{price_diff:,.0f} lower per sq ft",
            "★",
        )

    st.markdown("### Side-by-side breakdown")

    table = pd.DataFrame(
        {
            "": [
                "Location",
                "Type",
                "BHK / Bath",
                "Area",
                "Floor",
                "Age",
                "Furnishing",
                "Metro",
                "Estimate",
                "80% range",
                "₹ / sq ft",
            ],
            "Property A": [
                pa["location"],
                pa["property_type"],
                f"{pa['bhk']} / {pa['bathrooms']}",
                f"{pa['area_sqft']:,.0f} sq ft",
                f"{pa['floor']}/{pa['total_floors']}",
                f"{pa['age_years']:.1f}",
                pa["furnishing"],
                f"{pa['metro_dist_km']:.1f} km",
                fmt(listing, ra["estimate"]),
                f"{fmt(listing, ra['low'])} – {fmt(listing, ra['high'])}",
                f"{pps_a:,.0f}",
            ],
            "Property B": [
                pb["location"],
                pb["property_type"],
                f"{pb['bhk']} / {pb['bathrooms']}",
                f"{pb['area_sqft']:,.0f} sq ft",
                f"{pb['floor']}/{pb['total_floors']}",
                f"{pb['age_years']:.1f}",
                pb["furnishing"],
                f"{pb['metro_dist_km']:.1f} km",
                fmt(listing, rb["estimate"]),
                f"{fmt(listing, rb['low'])} – {fmt(listing, rb['high'])}",
                f"{pps_b:,.0f}",
            ],
        }
    )

    st.dataframe(table.astype(str), hide_index=True, width="stretch")

    chart_df = pd.DataFrame(
        {
            "Property": ["A", "B"],
            "₹ per sq ft": [pps_a, pps_b],
        }
    )

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("Property:N", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("₹ per sq ft:Q", title="Price per sq ft"),
            color=alt.Color(
                "Property:N",
                legend=None,
                scale=alt.Scale(range=["#5eead4", "#60a5fa"]),
            ),
            tooltip=["Property", "₹ per sq ft"],
        )
        .properties(height=300)
    )

    st.altair_chart(chart, width="stretch")


# ============================================================
# MARKET EXPLORER
# ============================================================
with tab_mkt:
    st.markdown('<div class="section-title">Mumbai micro-market explorer</div>', unsafe_allow_html=True)
    st.caption("Use this view to understand relative pricing across the simulated locations in the current dataset.")

    sub = df[df.listing_type == listing].copy()

    by_loc = (
        sub.groupby(["location", "city"], as_index=False)["price"]
        .median()
        .sort_values("price", ascending=False)
    )

    by_loc["label"] = [fmt(listing, v) for v in by_loc["price"]]

    chart = (
        alt.Chart(by_loc)
        .mark_bar(cornerRadiusEnd=7)
        .encode(
            y=alt.Y("location:N", sort="-x", title=None),
            x=alt.X("price:Q", title="Median price"),
            color=alt.Color(
                "city:N",
                title="City",
                scale=alt.Scale(range=["#5eead4", "#60a5fa", "#f4c95d"]),
            ),
            tooltip=["location", "city", "label"],
        )
        .properties(height=max(420, len(by_loc) * 30))
    )

    st.altair_chart(chart, width="stretch")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("### Area vs price")

        scatter = (
            alt.Chart(sub)
            .mark_circle(opacity=0.62, size=55)
            .encode(
                x=alt.X("area_sqft:Q", title="Area (sq ft)"),
                y=alt.Y("price:Q", title="Price"),
                color=alt.Color(
                    "bhk:O",
                    title="BHK",
                    scale=alt.Scale(range=["#60a5fa", "#5eead4", "#f4c95d"]),
                ),
                tooltip=[
                    "location",
                    "bhk",
                    "area_sqft",
                    "price",
                ],
            )
            .interactive()
            .properties(height=360)
        )

        st.altair_chart(scatter, width="stretch")

    with c2:
        st.markdown("### Price distribution by BHK")

        box = (
            alt.Chart(sub)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("bhk:O", title="BHK"),
                y=alt.Y("price:Q", title="Price"),
            )
            .properties(height=360)
        )

        st.altair_chart(box, width="stretch")

    st.markdown("### Dataset snapshot")

    s1, s2, s3, s4 = st.columns(4)

    s1.metric("Locations", sub["location"].nunique())
    s2.metric("Properties", f"{len(sub):,}")
    s3.metric("Median area", f"{sub['area_sqft'].median():,.0f} sq ft")
    s4.metric("Median BHK", f"{sub['bhk'].median():.0f}")


# ============================================================
# MODEL
# ============================================================
with tab_model:
    st.markdown('<div class="section-title">Model transparency</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <b>How the valuation works</b><br><br>
            Separate gradient-boosting regressors are trained for Sale and Rent.
            The pipeline uses log-price modelling plus engineered property,
            location and amenity features. Location target encoding is performed
            without leaking test information into the training process.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    for lt in ("Sale", "Rent"):
        b = models[lt]

        st.markdown(f"### {lt} model")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Test R²", f"{b.metrics['R2']:.3f}")
        c2.metric("MAPE", f"{b.metrics['MAPE_%']:.1f}%")
        c3.metric("MAE", fmt(lt, b.metrics["MAE"]))
        c4.metric(
            "Train / test",
            f"{b.metrics['train_rows']} / {b.metrics['test_rows']}",
        )

        imp = b.importances.head(10).reset_index()
        imp.columns = ["feature", "importance"]

        importance_chart = (
            alt.Chart(imp)
            .mark_bar(cornerRadiusEnd=5)
            .encode(
                y=alt.Y("feature:N", sort="-x", title=None),
                x=alt.X("importance:Q", title="Importance"),
            )
            .properties(height=300)
        )

        st.altair_chart(importance_chart, width="stretch")

    with st.expander("Preview model dataset"):
        st.dataframe(df.head(100), width="stretch")

    with st.expander("Important limitation"):
        st.write(
            "The current project uses a generated/simulated dataset. "
            "The model metrics describe performance on that dataset and should "
            "not be interpreted as real-world Mumbai property-market accuracy. "
            "For production use, replace the generator with a verified historical "
            "or licensed property dataset and retrain/evaluate the pipeline."
        )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        PropCompare · Mumbai & Navi Mumbai · ML-powered property analytics<br>
        Built for analytical exploration — not financial or real-estate advice.
    </div>
    """,
    unsafe_allow_html=True,
)
