import altair as alt
import pandas as pd
import streamlit as st

from propcompare.data import FURNISHING, LOCATIONS, PROPERTY_TYPES, clean, generate_dataset
from propcompare.model import predict, train_all
from propcompare.utils import fmt, per_sqft


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PropCompare — Mumbai Property Intelligence",
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
    --ink: #111827;
    --muted: #64748b;
    --line: rgba(15,23,42,.09);
    --card: rgba(255,255,255,.82);
    --accent: #2563eb;
    --accent2: #7c3aed;
    --green: #059669;
}

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(37,99,235,.10), transparent 28%),
        radial-gradient(circle at 92% 10%, rgba(124,58,237,.09), transparent 26%),
        linear-gradient(180deg, #f8fafc 0%, #ffffff 42%, #f8fafc 100%);
    color: var(--ink);
}

.block-container {
    max-width: 1440px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}

/* Header */
.hero {
    position: relative;
    overflow: hidden;
    padding: 34px 38px;
    margin-bottom: 24px;
    border: 1px solid var(--line);
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(255,255,255,.96), rgba(248,250,252,.78));
    box-shadow: 0 22px 70px rgba(15,23,42,.08);
    animation: heroIn .7s ease-out both;
}

.hero:before {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -70px;
    top: -110px;
    border-radius: 50%;
    background: linear-gradient(135deg, rgba(37,99,235,.18), rgba(124,58,237,.10));
    filter: blur(4px);
}

.hero-kicker {
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: .14em;
    font-size: .74rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero h1 {
    font-family: "Manrope", sans-serif;
    font-size: clamp(2rem, 4vw, 3.6rem);
    line-height: 1.02;
    letter-spacing: -.045em;
    margin: 0;
    position: relative;
}

.hero p {
    max-width: 760px;
    color: var(--muted);
    font-size: 1rem;
    margin: 15px 0 0;
    position: relative;
}

/* Cards */
.metric-card {
    padding: 20px 22px;
    border-radius: 20px;
    background: var(--card);
    border: 1px solid var(--line);
    box-shadow: 0 12px 35px rgba(15,23,42,.06);
    transition: transform .25s ease, box-shadow .25s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 45px rgba(15,23,42,.11);
}

.metric-label {
    color: var(--muted);
    font-size: .78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.metric-value {
    font-family: "Manrope", sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    margin-top: 7px;
}

.metric-sub {
    color: var(--muted);
    font-size: .82rem;
    margin-top: 4px;
}

/* Section titles */
.section-title {
    font-family: "Manrope", sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -.025em;
    margin: 10px 0 3px;
}

.section-sub {
    color: var(--muted);
    font-size: .88rem;
    margin-bottom: 14px;
}

/* Comparison */
.compare-card {
    padding: 24px;
    border-radius: 24px;
    background: rgba(255,255,255,.86);
    border: 1px solid var(--line);
    box-shadow: 0 16px 45px rgba(15,23,42,.065);
    transition: transform .25s ease, box-shadow .25s ease;
}

.compare-card:hover {
    transform: translateY(-5px) perspective(900px) rotateX(.7deg);
    box-shadow: 0 25px 60px rgba(15,23,42,.11);
}

.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    background: #eff6ff;
    color: #1d4ed8;
    font-size: .72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .07em;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 700 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #2563eb !important;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] > div {
    border-radius: 12px !important;
}

div[data-testid="stNumberInput"] input {
    border-radius: 12px;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: transparent;
}

/* Scroll-in utility */
.reveal {
    animation: reveal .65s ease-out both;
}

@keyframes reveal {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes heroIn {
    from { opacity: 0; transform: translateY(-14px) scale(.99); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

/* Mobile */
@media (max-width: 800px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .hero {
        padding: 26px 22px;
        border-radius: 22px;
    }
}
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


@st.cache_resource(show_spinner="Training property models…")
def load_models():
    return train_all(load_data())


df = load_data()
models = load_models()
LOCS = sorted(LOCATIONS)


# ============================================================
# HELPERS
# ============================================================
def property_form(key: str, defaults: dict) -> dict:
    """Render one property configuration and return model features."""

    st.markdown('<div class="section-title">Property details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Configure the property characteristics used by the estimator.</div>',
        unsafe_allow_html=True,
    )

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

    st.caption("Amenities")

    a1, a2, a3, a4 = st.columns(4)

    lift = int(a1.checkbox("Lift", defaults["lift"], key=f"{key}_lift"))
    gym = int(a2.checkbox("Gym", defaults["gym"], key=f"{key}_gym"))
    pool = int(a3.checkbox("Pool", defaults["pool"], key=f"{key}_pool"))
    sec = int(
        a4.checkbox("24×7 security", defaults["security"], key=f"{key}_sec")
    )

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


def metric_card(label, value, sub=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """


def property_score(prop: dict) -> int:
    """Simple explainable presentation score, not the ML prediction."""
    score = 50

    score += min(prop["parking"] * 6, 12)
    score += min(prop["lift"] * 5, 5)
    score += min(prop["gym"] * 5, 5)
    score += min(prop["pool"] * 4, 4)
    score += min(prop["security"] * 5, 5)

    score += max(0, 10 - prop["metro_dist_km"] * 4)
    score += max(0, 5 - prop["school_dist_km"])

    score -= min(prop["age_years"] * .25, 8)

    return max(0, min(100, round(score)))


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
<div class="hero reveal">
    <div class="hero-kicker">Property intelligence · Mumbai Metropolitan Region</div>
    <h1>Compare properties with<br>data, not guesswork.</h1>
    <p>
        Estimate sale or rental value, compare micro-markets, understand
        price-per-square-foot and inspect the model behind every estimate.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# GLOBAL MODE
# ============================================================
listing = st.radio(
    "Market mode",
    ["Sale", "Rent"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown(
    f"""
<div class="badge">{listing} intelligence</div>
""",
    unsafe_allow_html=True,
)

tab_pred, tab_cmp, tab_mkt, tab_model = st.tabs(
    [
        "💰  Estimate",
        "⚖️  Compare",
        "📊  Market",
        "🧠  Model",
    ]
)


# ============================================================
# ESTIMATE
# ============================================================
with tab_pred:
    left, right = st.columns([1.45, 1], gap="large")

    with left:
        prop = property_form("single", A_DEF)

    result = predict(models, listing, prop)
    pps = per_sqft(listing, result["estimate"], prop["area_sqft"])

    with right:
        st.markdown(
            '<div class="section-title">Valuation snapshot</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            metric_card(
                "Estimated " + ("sale value" if listing == "Sale" else "monthly rent"),
                fmt(listing, result["estimate"]),
                f"≈ ₹{pps:,.0f} per sq ft",
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                metric_card(
                    "80% estimated range",
                    f"{fmt(listing, result['low'])} – {fmt(listing, result['high'])}",
                    "Model uncertainty band",
                ),
                unsafe_allow_html=True,
            )

        with c2:
            loc_sub = df[
                (df.location == prop["location"])
                & (df.listing_type == listing)
            ]

            median = loc_sub["price"].median() if len(loc_sub) else None

            st.markdown(
                metric_card(
                    "Local median",
                    fmt(listing, median) if median else "N/A",
                    f"{prop['location']} market sample",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        score = property_score(prop)

        st.markdown(
            metric_card(
                "Property profile score",
                f"{score}/100",
                "Explainable amenity + accessibility indicator",
            ),
            unsafe_allow_html=True,
        )

        st.info(
            "This score is a presentation metric, not the model's prediction. "
            "The valuation is produced by the trained regression model."
        )


# ============================================================
# COMPARISON
# ============================================================
with tab_cmp:
    st.markdown(
        '<div class="section-title">Side-by-side property intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Change either property to see how location, size, age and amenities affect the estimate.</div>',
        unsafe_allow_html=True,
    )

    ca, cb = st.columns(2, gap="large")

    with ca:
        st.markdown('<div class="badge">Property A</div>', unsafe_allow_html=True)
        pa = property_form("A", A_DEF)

    with cb:
        st.markdown('<div class="badge">Property B</div>', unsafe_allow_html=True)
        pb = property_form("B", B_DEF)

    ra = predict(models, listing, pa)
    rb = predict(models, listing, pb)

    pps_a = per_sqft(listing, ra["estimate"], pa["area_sqft"])
    pps_b = per_sqft(listing, rb["estimate"], pb["area_sqft"])

    st.divider()

    m1, m2, m3, m4 = st.columns(4)

    m1.markdown(
        metric_card(
            "Property A",
            fmt(listing, ra["estimate"]),
            f"{pa['location']} · ₹{pps_a:,.0f}/sq ft",
        ),
        unsafe_allow_html=True,
    )

    m2.markdown(
        metric_card(
            "Property B",
            fmt(listing, rb["estimate"]),
            f"{pb['location']} · ₹{pps_b:,.0f}/sq ft",
        ),
        unsafe_allow_html=True,
    )

    diff_pct = (rb["estimate"] / ra["estimate"] - 1) * 100

    m3.markdown(
        metric_card(
            "Price difference",
            f"{diff_pct:+.1f}%",
            "B relative to A",
        ),
        unsafe_allow_html=True,
    )

    better = "A" if pps_a < pps_b else "B"

    m4.markdown(
        metric_card(
            "Value leader",
            f"Property {better}",
            "Lower estimated ₹/sq ft",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

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
                f"{pa['floor']} / {pa['total_floors']}",
                f"{pa['age_years']:.1f}",
                pa["furnishing"],
                f"{pa['metro_dist_km']:.1f} km",
                fmt(listing, ra["estimate"]),
                f"{fmt(listing, ra['low'])} – {fmt(listing, ra['high'])}",
                f"₹{pps_a:,.0f}",
            ],
            "Property B": [
                pb["location"],
                pb["property_type"],
                f"{pb['bhk']} / {pb['bathrooms']}",
                f"{pb['area_sqft']:,.0f} sq ft",
                f"{pb['floor']} / {pb['total_floors']}",
                f"{pb['age_years']:.1f}",
                pb["furnishing"],
                f"{pb['metro_dist_km']:.1f} km",
                fmt(listing, rb["estimate"]),
                f"{fmt(listing, rb['low'])} – {fmt(listing, rb['high'])}",
                f"₹{pps_b:,.0f}",
            ],
        }
    )

    st.dataframe(
        table.astype(str),
        hide_index=True,
        width="stretch",
    )

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
            y=alt.Y("₹ per sq ft:Q", title="Estimated ₹ / sq ft"),
            tooltip=["Property", "₹ per sq ft"],
        )
        .properties(height=300, title="Price efficiency")
    )

    st.altair_chart(chart, width="stretch")


# ============================================================
# MARKET EXPLORER
# ============================================================
with tab_mkt:
    sub = df[df.listing_type == listing].copy()

    st.markdown(
        '<div class="section-title">Mumbai micro-market explorer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Explore how the simulated market sample varies across locations, property sizes and BHK categories.</div>',
        unsafe_allow_html=True,
    )

    by_loc = (
        sub.groupby(["location", "city"], as_index=False)["price"]
        .median()
        .sort_values("price", ascending=False)
    )

    by_loc["label"] = [fmt(listing, v) for v in by_loc["price"]]

    market_chart = (
        alt.Chart(by_loc)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            y=alt.Y("location:N", sort="-x", title=None),
            x=alt.X("price:Q", title="Median price"),
            color=alt.Color("city:N", title="Market"),
            tooltip=["location", "city", "label"],
        )
        .properties(height=560)
    )

    st.altair_chart(market_chart, width="stretch")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            '<div class="section-title">Area vs price</div>',
            unsafe_allow_html=True,
        )

        scatter = (
            alt.Chart(sub)
            .mark_circle(opacity=0.55, size=55)
            .encode(
                x=alt.X("area_sqft:Q", title="Area (sq ft)"),
                y=alt.Y("price:Q", title="Price"),
                color=alt.Color("bhk:O", title="BHK"),
                tooltip=[
                    "location",
                    "bhk",
                    "area_sqft",
                    "price",
                ],
            )
            .interactive()
            .properties(height=400)
        )

        st.altair_chart(scatter, width="stretch")

    with c2:
        st.markdown(
            '<div class="section-title">Price distribution by BHK</div>',
            unsafe_allow_html=True,
        )

        box = (
            alt.Chart(sub)
            .mark_boxplot(size=55)
            .encode(
                x=alt.X("bhk:O", title="BHK"),
                y=alt.Y("price:Q", title="Price"),
            )
            .properties(height=400)
        )

        st.altair_chart(box, width="stretch")


# ============================================================
# MODEL TRANSPARENCY
# ============================================================
with tab_model:
    st.markdown(
        '<div class="section-title">Model transparency</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Performance and feature importance for the Sale and Rent estimators.</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "Important: the current application trains on a generated/simulated dataset. "
        "Model metrics therefore describe performance on that dataset and should not "
        "be interpreted as real-world Mumbai valuation accuracy."
    )

    for lt in ("Sale", "Rent"):
        b = models[lt]

        st.markdown(
            f'<div class="badge">{lt} model</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.markdown(
            metric_card("R²", f"{b.metrics['R2']:.3f}", "Test set"),
            unsafe_allow_html=True,
        )
        c2.markdown(
            metric_card("MAPE", f"{b.metrics['MAPE_%']:.1f}%", "Mean absolute % error"),
            unsafe_allow_html=True,
        )
        c3.markdown(
            metric_card("MAE", fmt(lt, b.metrics["MAE"]), "Mean absolute error"),
            unsafe_allow_html=True,
        )
        c4.markdown(
            metric_card(
                "Rows",
                f"{b.metrics['train_rows']:,} / {b.metrics['test_rows']:,}",
                "Train / test",
            ),
            unsafe_allow_html=True,
        )

        imp = b.importances.head(10).reset_index()
        imp.columns = ["feature", "importance"]

        feature_chart = (
            alt.Chart(imp)
            .mark_bar(cornerRadiusEnd=5)
            .encode(
                y=alt.Y("feature:N", sort="-x", title=None),
                x=alt.X("importance:Q", title="Importance"),
                tooltip=["feature", "importance"],
            )
            .properties(height=300)
        )

        st.altair_chart(feature_chart, width="stretch")

    with st.expander("Preview generated dataset"):
        st.dataframe(df.head(100), width="stretch")


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
<hr style="border:0;border-top:1px solid rgba(15,23,42,.08);margin-top:50px;">

<div style="text-align:center;color:#64748b;font-size:.8rem;padding:15px 0;">
    <strong>PropCompare</strong> · Mumbai & Navi Mumbai Property Intelligence
    <br>
    Estimates are indicative and generated from the application's current model/data.
    They are not property valuations, offers, or financial advice.
</div>
""",
    unsafe_allow_html=True,
)
