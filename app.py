"""PropCompare — Mumbai & Navi Mumbai property estimates.

Run with:  streamlit run app.py
Needs Streamlit >= 1.50 (fragments, st.container(key=...), width="stretch").
"""

from __future__ import annotations

from html import escape

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
    page_title="PropCompare — Mumbai home values, explained",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Palette (kept in one place so charts and CSS agree)
INK = "#12262B"
MUTED = "#4B6168"
SEA = "#17646E"
MARIGOLD = "#E39B2B"
PLUM = "#7A4E8C"
GRID = "#DCE5E3"
CHART_RANGE = ["#17646E", "#E39B2B", "#7A4E8C", "#3E8E5A", "#C4573A", "#4C7BD9"]

alt.data_transformers.disable_max_rows()


# ============================================================
# STYLES
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Figtree:wght@400;500;600;700&display=swap');

:root {
    --paper: #eef3f1;
    --surface: #ffffff;
    --ink: #12262b;
    --muted: #4b6168;
    --line: #d5dedc;
    --sea: #17646e;
    --sea-deep: #0e3238;
    --sea-soft: #d8ebea;
    --marigold: #e39b2b;
    --marigold-soft: #fbebcb;
    --ease: cubic-bezier(.2, .7, .2, 1);
}

/* ---------- Base & smooth scrolling ---------- */
html,
[data-testid="stMain"],
section.main,
.main {
    scroll-behavior: smooth;
    scroll-padding-top: 6rem;
}

html, body, [class*="css"], .stApp {
    font-family: "Figtree", system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink);
}

.stApp {
    background:
        radial-gradient(900px 420px at 92% -4%, rgba(227,155,43,.16), transparent 70%),
        radial-gradient(800px 500px at -5% 8%, rgba(23,100,110,.10), transparent 70%),
        var(--paper);
}

.block-container {
    max-width: 1280px;
    padding: 1rem 2.5rem 4rem;
}

h1, h2, h3, h4,
.section-title, .hero h1 {
    font-family: "Bricolage Grotesque", "Figtree", system-ui, sans-serif;
    letter-spacing: -.02em;
    color: var(--ink);
}

/* ---------- Accessibility ---------- */
:focus-visible {
    outline: 3px solid var(--marigold) !important;
    outline-offset: 2px !important;
    border-radius: 8px;
}

.skip-link {
    position: absolute;
    left: -9999px;
    top: .5rem;
    background: var(--ink);
    color: #fff !important;
    padding: .6rem 1rem;
    border-radius: 10px;
    z-index: 1000;
}
.skip-link:focus { left: .75rem; }

/* ---------- Sticky top bar ---------- */
.st-key-topbar {
    position: sticky;
    top: 0;
    z-index: 60;
    margin: 0 -1rem 1rem;
    padding: .55rem 1rem;
    background: rgba(238, 243, 241, .82);
    -webkit-backdrop-filter: blur(14px) saturate(1.2);
    backdrop-filter: blur(14px) saturate(1.2);
    border-bottom: 1px solid var(--line);
}

.brand {
    display: flex;
    align-items: center;
    gap: .65rem;
    font-family: "Bricolage Grotesque", sans-serif;
    font-weight: 800;
    font-size: 1.2rem;
    letter-spacing: -.02em;
}
.brand-mark {
    width: 30px; height: 30px;
    border-radius: 9px;
    background: linear-gradient(145deg, var(--sea), var(--sea-deep));
    display: grid; place-items: center;
}
.brand-mark svg { width: 18px; height: 18px; }
.brand small {
    font-family: "Figtree", sans-serif;
    font-weight: 500;
    font-size: .85rem;
    color: var(--muted);
    margin-left: .15rem;
}

/* Market toggle as pills */
div[role="radiogroup"] { gap: .4rem; justify-content: flex-end; }
div[role="radiogroup"] label {
    border: 1px solid var(--line);
    background: var(--surface);
    border-radius: 999px;
    padding: .3rem .95rem;
    transition: background .2s var(--ease), border-color .2s var(--ease);
}
div[role="radiogroup"] label:hover { border-color: var(--sea); }
div[role="radiogroup"] label:has(input:checked) {
    background: var(--sea);
    border-color: var(--sea);
}
div[role="radiogroup"] label:has(input:checked) p { color: #fff; }

/* ---------- Hero (one orchestrated moment) ---------- */
.hero {
    display: grid;
    grid-template-columns: 1.35fr .85fr;
    align-items: center;
    gap: 1rem;
    padding: 2.4rem 2.6rem;
    border-radius: 30px;
    background:
        linear-gradient(120deg, rgba(255,255,255,.92), rgba(255,255,255,.62)),
        var(--surface);
    border: 1px solid var(--line);
    box-shadow: 0 30px 60px -40px rgba(14, 50, 56, .45);
    overflow: hidden;
}
.hero-copy > * { animation: rise .8s var(--ease) both; }
.hero-copy > *:nth-child(2) { animation-delay: .08s; }
.hero-copy > *:nth-child(3) { animation-delay: .16s; }
.hero-copy > *:nth-child(4) { animation-delay: .24s; }

.hero h1 {
    font-size: clamp(2rem, 4.2vw, 3.5rem);
    line-height: 1.04;
    font-weight: 800;
    margin: 0 0 .9rem;
}
.hero p {
    max-width: 52ch;
    color: var(--muted);
    font-size: 1.08rem;
    line-height: 1.65;
    margin: 0 0 1.2rem;
}
.data-note {
    display: inline-flex;
    align-items: center;
    gap: .55rem;
    padding: .5rem .9rem;
    border-radius: 999px;
    background: var(--marigold-soft);
    color: #6b4a0b;
    font-size: .88rem;
    font-weight: 500;
}
.data-note b { font-weight: 700; }

@keyframes rise {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: none; }
}

/* Isometric skyline — pure CSS 3D, decorative */
.scene-wrap {
    position: relative;
    height: 340px;
    perspective: 1400px;
}
.sun {
    position: absolute;
    right: 12%; top: 4%;
    width: 110px; height: 110px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #f6c46a, var(--marigold));
    opacity: .9;
}
.scene {
    position: absolute;
    left: 50%; top: 60%;
    width: 192px; height: 192px;
    margin: -96px 0 0 -96px;
    transform-style: preserve-3d;
    transform: rotateX(58deg) rotateZ(42deg);
    animation: sway 16s ease-in-out infinite alternate;
    will-change: transform;
}
@keyframes sway {
    from { transform: rotateX(58deg) rotateZ(38deg); }
    to   { transform: rotateX(58deg) rotateZ(48deg); }
}
.ground {
    position: absolute;
    inset: -34px;
    border-radius: 16px;
    background:
        linear-gradient(rgba(23,100,110,.12) 1px, transparent 1px) 0 0 / 24px 24px,
        linear-gradient(90deg, rgba(23,100,110,.12) 1px, transparent 1px) 0 0 / 24px 24px,
        var(--sea-soft);
    transform: translateZ(-1px);
}
.bld {
    position: absolute;
    left: var(--x); top: var(--y);
    width: 52px; height: 52px;
    transform-style: preserve-3d;
}
.bld i { position: absolute; display: block; }
.bld .top  { inset: 0; transform: translateZ(var(--h)); background: var(--c-top); }
.bld .fy {
    left: 0; top: 100%;
    width: 100%; height: var(--h);
    transform-origin: top;
    transform: rotateX(90deg);
    background-color: var(--c-front);
    background-image:
        repeating-linear-gradient(90deg, transparent 0 8px, rgba(255,255,255,.26) 8px 12px),
        repeating-linear-gradient(0deg, transparent 0 10px, rgba(7,30,36,.16) 10px 11px);
}
.bld .fx {
    left: 100%; top: 0;
    width: var(--h); height: 100%;
    transform-origin: left;
    transform: rotateY(-90deg);
    background-color: var(--c-side);
    background-image:
        repeating-linear-gradient(0deg, transparent 0 8px, rgba(255,255,255,.2) 8px 12px),
        repeating-linear-gradient(90deg, transparent 0 10px, rgba(7,30,36,.18) 10px 11px);
}
.t-a { --c-top: #d8ebea; --c-front: #2f8a93; --c-side: #1b5f68; }
.t-b { --c-top: #fbebcb; --c-front: #e8b357; --c-side: #c58a2a; }
.t-c { --c-top: #e6eceb; --c-front: #7fa5ab; --c-side: #5b8189; }

/* ---------- Section headings ---------- */
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin: .4rem 0 .15rem;
}
.section-sub {
    color: var(--muted);
    margin: 0 0 1rem;
    max-width: 62ch;
}
.group-title {
    font-weight: 700;
    font-size: 1rem;
    margin: 1.3rem 0 .3rem;
    padding-bottom: .35rem;
    border-bottom: 1px solid var(--line);
}

/* ---------- Result cards ---------- */
.valuation {
    background: linear-gradient(160deg, #134c55, var(--sea-deep));
    color: #f2f8f7;
    border-radius: 24px;
    padding: 1.5rem 1.6rem 1.35rem;
    box-shadow: 0 24px 44px -24px rgba(14, 50, 56, .6);
}
.valuation .label { color: #b9d6d4; font-size: .95rem; font-weight: 500; }
.valuation .value {
    font-family: "Bricolage Grotesque", sans-serif;
    font-weight: 800;
    font-size: clamp(2rem, 3.4vw, 2.9rem);
    letter-spacing: -.03em;
    line-height: 1.1;
    margin: .25rem 0 .2rem;
}
.valuation .sub { color: #b9d6d4; font-size: .92rem; }

.meter { margin-top: 1.1rem; }
.meter-track {
    position: relative;
    height: 8px;
    border-radius: 99px;
    background: linear-gradient(90deg, rgba(255,255,255,.16), rgba(227,155,43,.5));
}
.meter-dot {
    position: absolute;
    top: 50%;
    width: 20px; height: 20px;
    margin: -10px 0 0 -10px;
    border-radius: 50%;
    background: var(--marigold);
    border: 3px solid #fff;
    box-shadow: 0 2px 8px rgba(0,0,0,.3);
}
.meter-ends {
    display: flex;
    justify-content: space-between;
    margin-top: .55rem;
    font-size: .85rem;
    color: #d3e6e4;
}

.stat {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1rem 1.15rem;
    height: 100%;
}
.stat .label { color: var(--muted); font-size: .9rem; font-weight: 500; }
.stat .value {
    font-family: "Bricolage Grotesque", sans-serif;
    font-weight: 700;
    font-size: 1.45rem;
    letter-spacing: -.02em;
    margin-top: .2rem;
}
.stat .sub { color: var(--muted); font-size: .88rem; margin-top: .2rem; }
.stat.good { border-color: var(--sea); background: #f2faf9; }

.note {
    background: var(--surface);
    border-left: 4px solid var(--marigold);
    border-radius: 12px;
    padding: .85rem 1.1rem;
    color: var(--ink);
    line-height: 1.6;
}

/* Subtle 3D tilt on the numbers people care about */
.tilt {
    transition: transform .4s var(--ease), box-shadow .4s var(--ease);
    transform: perspective(900px) rotateX(0) rotateY(0);
}
.tilt:hover {
    transform: perspective(900px) rotateX(2.5deg) rotateY(-3deg) translateY(-3px);
}

.jump { display: none; }

/* ---------- Streamlit widgets ---------- */
[data-testid="stWidgetLabel"] p { font-weight: 600; font-size: .9rem; color: var(--ink); }

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
}

div.stButton > button,
div.stDownloadButton > button {
    border-radius: 999px;
    border: 1px solid var(--line);
    background: var(--surface);
    color: var(--ink);
    font-weight: 600;
    padding: .4rem 1.1rem;
    transition: transform .2s var(--ease), border-color .2s var(--ease), box-shadow .2s var(--ease);
}
div.stButton > button:hover,
div.stDownloadButton > button:hover {
    border-color: var(--sea);
    color: var(--sea);
    transform: translateY(-1px);
    box-shadow: 0 8px 18px -12px rgba(14,50,56,.5);
}

button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 1rem;
    color: var(--muted);
    padding: .7rem 1rem;
}
button[data-baseweb="tab"][aria-selected="true"] { color: var(--sea); }
div[data-baseweb="tab-highlight"] { background: var(--marigold) !important; height: 3px !important; }
div[data-baseweb="tab-border"] { background: var(--line) !important; }

div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
}
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: .8rem 1rem;
}
[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    background: var(--surface);
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: var(--muted);
    font-size: .88rem;
    line-height: 1.7;
    padding: 2.5rem 0 .5rem;
}
.footer a { color: var(--sea); font-weight: 600; text-decoration: none; }
.footer a:hover { text-decoration: underline; }

/* Hide Streamlit chrome we don't need */
#MainMenu { visibility: hidden; }
[data-testid="stFooter"], .stApp > footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- Responsive ---------- */
@media (max-width: 900px) {
    .block-container { padding: .5rem 1rem 3rem; }
    .hero { grid-template-columns: 1fr; padding: 1.6rem 1.4rem; border-radius: 22px; }
    .scene-wrap { display: none; }
    .jump {
        display: inline-block;
        margin: .4rem 0 .2rem;
        padding: .45rem .95rem;
        border-radius: 999px;
        background: var(--sea);
        color: #fff !important;
        font-weight: 600;
        text-decoration: none;
    }
    .st-key-topbar { margin: 0 -.5rem .75rem; }
}

/* ---------- Respect motion preferences ---------- */
@media (prefers-reduced-motion: reduce) {
    html, [data-testid="stMain"], section.main, .main { scroll-behavior: auto; }
    *, *::before, *::after {
        animation: none !important;
        transition-duration: .01ms !important;
    }
    .tilt:hover { transform: none; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# DATA & MODELS
# ============================================================
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return clean(generate_dataset())


@st.cache_resource(show_spinner="Getting the pricing models ready…")
def load_models():
    return train_all(load_data())


try:
    df = load_data()
    models = load_models()
except Exception as exc:  # noqa: BLE001 — show a readable message instead of a traceback
    st.error(
        "We couldn't load the pricing data or models. Check that the `propcompare` "
        "package is installed, then refresh the page."
    )
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()

LOCS = sorted(LOCATIONS)


@st.cache_data(show_spinner=False, max_entries=512)
def _predict_cached(listing: str, prop_items: tuple, _models) -> dict:
    """Memoise predictions so moving between tabs doesn't recompute them."""
    return predict(_models, listing, dict(prop_items))


def estimate(listing: str, prop: dict) -> dict:
    return _predict_cached(listing, tuple(sorted(prop.items())), models)


# ============================================================
# HELPERS
# ============================================================
FORM_SUFFIXES = [
    "loc", "pt", "bhk", "bath", "area", "tf", "fl", "age",
    "fu", "pk", "mt", "sc", "lift", "gym", "pool", "sec",
]

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

COMPACT_DEF = dict(
    A_DEF,
    bhk=1,
    bathrooms=1,
    area_sqft=480,
    floor=4,
    total_floors=12,
    age_years=8.0,
    furn_idx=0,
    parking=0,
    pool=False,
    gym=False,
)

PRESETS = {
    "2 BHK, Andheri West": A_DEF,
    "Roomy 2 BHK, Kharghar": B_DEF,
    "Compact 1 BHK": COMPACT_DEF,
}


def _furn_value(idx: int) -> str:
    return FURNISHING[max(0, min(int(idx), len(FURNISHING) - 1))]


def _apply_preset(key: str, d: dict) -> None:
    """Button callback: write a preset into the form's widget state."""
    values = {
        "loc": d["location"] if d["location"] in LOCS else LOCS[0],
        "pt": d["property_type"],
        "bhk": int(d["bhk"]),
        "bath": int(d["bathrooms"]),
        "area": int(d["area_sqft"]),
        "tf": int(d["total_floors"]),
        "fl": int(d["floor"]),
        "age": float(d["age_years"]),
        "fu": _furn_value(d["furn_idx"]),
        "pk": int(d["parking"]),
        "mt": float(d["metro_dist_km"]),
        "sc": float(d["school_dist_km"]),
        "lift": bool(d["lift"]),
        "gym": bool(d["gym"]),
        "pool": bool(d["pool"]),
        "sec": bool(d["security"]),
    }
    for suffix, value in values.items():
        st.session_state[f"{key}_{suffix}"] = value


def _swap_ab() -> None:
    """Button callback: swap every input between Property A and B."""
    for s in FORM_SUFFIXES:
        a, b = st.session_state.get(f"A_{s}"), st.session_state.get(f"B_{s}")
        if a is not None and b is not None:
            st.session_state[f"A_{s}"], st.session_state[f"B_{s}"] = b, a


def stat_card(label: str, value: str, sub: str = "", *, good: bool = False, tilt: bool = True) -> None:
    classes = "stat" + (" good" if good else "") + (" tilt" if tilt else "")
    st.markdown(
        f"""
        <div class="{classes}">
            <div class="label">{escape(label)}</div>
            <div class="value">{escape(value)}</div>
            <div class="sub">{escape(sub)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def range_meter(listing: str, low: float, est: float, high: float) -> str:
    span = high - low
    pos = 50 if span <= 0 else max(0, min(100, (est - low) / span * 100))
    label = (
        f"Estimate {fmt(listing, est)}. Likely range {fmt(listing, low)} to {fmt(listing, high)}."
    )
    return f"""
    <div class="meter" role="img" aria-label="{escape(label)}">
        <div class="meter-track"><span class="meter-dot" style="left:{pos:.1f}%"></span></div>
        <div class="meter-ends"><span>{escape(fmt(listing, low))}</span><span>{escape(fmt(listing, high))}</span></div>
    </div>
    """


def axis_scale(listing: str) -> tuple[float, str]:
    """Readable axis units: crore for sale, thousand for rent."""
    return (1e7, "Price (₹ crore)") if listing == "Sale" else (1e3, "Rent (₹ thousand / month)")


def style_chart(chart: alt.Chart | alt.LayerChart) -> alt.Chart | alt.LayerChart:
    return (
        chart.configure(background="transparent", font="Figtree")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=MUTED, titleColor=INK, gridColor=GRID, domainColor="#B9C7C4",
            tickColor="#B9C7C4", labelFontSize=12, titleFontSize=12, titleFontWeight=600,
        )
        .configure_legend(labelColor=MUTED, titleColor=INK, labelFontSize=12, titleFontSize=12)
    )


def show_chart(chart) -> None:
    st.altair_chart(style_chart(chart), width="stretch", theme=None)


def property_form(key: str, defaults: dict, *, presets: bool = False) -> dict:
    """Render the property inputs and return model features."""

    if presets:
        st.caption("Start from an example, then adjust anything you like.")
        cols = st.columns(len(PRESETS))
        for col, (name, preset) in zip(cols, PRESETS.items()):
            col.button(name, key=f"{key}_preset_{name}", on_click=_apply_preset, args=(key, preset), width="stretch")

    st.markdown('<div class="group-title">The home</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    loc = c1.selectbox(
        "Neighbourhood", LOCS, index=LOCS.index(defaults["location"]), key=f"{key}_loc",
        help="Start typing to search.",
    )
    ptype = c2.selectbox(
        "Property type", PROPERTY_TYPES, index=PROPERTY_TYPES.index(defaults["property_type"]), key=f"{key}_pt",
    )

    c1, c2, c3 = st.columns(3)
    bhk = c1.number_input("Bedrooms (BHK)", 1, 6, defaults["bhk"], key=f"{key}_bhk")
    bath = c2.number_input("Bathrooms", 1, 8, defaults["bathrooms"], key=f"{key}_bath")
    area = c3.number_input(
        "Carpet area (sq ft)", 250, 6000, int(defaults["area_sqft"]), step=25, key=f"{key}_area",
    )

    per_bed = area / max(int(bhk), 1)
    if per_bed < 220:
        st.info(f"{int(bhk)} BHK in {int(area):,} sq ft is unusually tight, so the estimate may be less reliable.")
    elif per_bed > 2500:
        st.info(f"{int(area):,} sq ft for {int(bhk)} BHK is unusually spacious, so the estimate may be less reliable.")

    st.markdown('<div class="group-title">The building</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    total = c1.number_input("Floors in building", 1, 60, defaults["total_floors"], key=f"{key}_tf")

    # Keep the floor input valid if the building height was lowered.
    if st.session_state.get(f"{key}_fl", 0) > int(total):
        st.session_state[f"{key}_fl"] = int(total)
    floor = c2.number_input(
        "Floor", 0, int(total), min(defaults["floor"], int(total)), key=f"{key}_fl",
        help="0 is ground floor.",
    )
    age = c3.number_input(
        "Age (years)", 0.0, 50.0, float(defaults["age_years"]), step=0.5, key=f"{key}_age",
    )

    c1, c2 = st.columns(2)
    furn = c1.selectbox(
        "Furnishing", FURNISHING, index=max(0, min(defaults["furn_idx"], len(FURNISHING) - 1)), key=f"{key}_fu",
    )
    parking = c2.selectbox("Parking spots", [0, 1, 2], index=defaults["parking"], key=f"{key}_pk")

    st.markdown('<div class="group-title">Around it</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    metro = c1.number_input(
        "Distance to metro (km)", 0.0, 15.0, float(defaults["metro_dist_km"]), step=0.1, key=f"{key}_mt",
    )
    school = c2.slider(
        "Distance to school (km)", 0.3, 5.0, float(defaults["school_dist_km"]), 0.1, key=f"{key}_sc",
    )

    st.markdown('<div class="group-title">Amenities</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    lift = int(a1.checkbox("Lift", defaults["lift"], key=f"{key}_lift"))
    gym = int(a2.checkbox("Gym", defaults["gym"], key=f"{key}_gym"))
    pool = int(a3.checkbox("Pool", defaults["pool"], key=f"{key}_pool"))
    sec = int(a4.checkbox("24×7 security", defaults["security"], key=f"{key}_sec"))

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


# ============================================================
# TOP BAR
# ============================================================
st.markdown('<span id="top"></span><a class="skip-link" href="#main-content">Skip to content</a>', unsafe_allow_html=True)

with st.container(key="topbar"):
    left, right = st.columns([3, 2], vertical_alignment="center")
    left.markdown(
        """
        <div class="brand">
            <span class="brand-mark" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M4 21V9l6-4v16M10 21V3l10 5v13M2 21h20"/>
                </svg>
            </span>
            PropCompare <small>Mumbai &amp; Navi Mumbai</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    listing = right.radio(
        "Show prices for",
        ["Sale", "Rent"],
        format_func=lambda v: "Buying" if v == "Sale" else "Renting",
        horizontal=True,
        label_visibility="collapsed",
        key="listing_mode",
    )


# ============================================================
# HERO
# ============================================================
BUILDINGS = [
    (0, 0, 190, "a"), (70, 0, 130, "b"), (140, 0, 86, "c"),
    (0, 70, 120, "c"), (70, 70, 160, "a"), (140, 70, 70, "b"),
    (0, 140, 64, "b"), (70, 140, 92, "c"), (140, 140, 44, "a"),
]
skyline = "".join(
    f'<div class="bld t-{tone}" style="--x:{x}px;--y:{y}px;--h:{h}px">'
    '<i class="top"></i><i class="fy"></i><i class="fx"></i></div>'
    for x, y, h, tone in BUILDINGS
)

st.markdown(
    f"""
    <section class="hero" aria-labelledby="hero-title">
        <div class="hero-copy">
            <h1 id="hero-title">Know what a Mumbai home is worth before you talk price.</h1>
            <p>
                Get a quick estimate for buying or renting, set two homes side by side,
                and see how neighbourhoods compare. Every number comes with a likely range,
                so you know how much to trust it.
            </p>
            <span class="data-note" role="note">
                <b>Demo data.</b> Built on simulated listings, so treat results as a guide.
            </span>
        </div>
        <div class="scene-wrap" aria-hidden="true">
            <div class="sun"></div>
            <div class="scene"><div class="ground"></div>{skyline}</div>
        </div>
    </section>
    <div id="main-content" tabindex="-1"></div>
    """,
    unsafe_allow_html=True,
)

st.write("")

tab_est, tab_cmp, tab_mkt, tab_model = st.tabs(
    ["Estimate", "Compare two homes", "Neighbourhoods", "How it works"]
)


# ============================================================
# ESTIMATE
# ============================================================
@st.fragment
def estimate_view(listing: str) -> None:
    form_col, result_col = st.columns([1.45, 1], gap="large")

    with form_col:
        st.markdown('<div class="section-title">Describe the home</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-sub">Fill in what you know. The estimate updates as you go.</p>'
            '<a class="jump" href="#estimate-result">Jump to the estimate</a>',
            unsafe_allow_html=True,
        )
        prop = property_form("single", A_DEF, presets=True)

    result = estimate(listing, prop)
    pps = per_sqft(listing, result["estimate"], prop["area_sqft"])
    noun = "sale price" if listing == "Sale" else "monthly rent"

    with result_col:
        st.markdown('<div id="estimate-result"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Your estimate</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="valuation tilt" role="status" aria-live="polite">
                <div class="label">Estimated {noun}</div>
                <div class="value">{escape(fmt(listing, result["estimate"]))}</div>
                <div class="sub">≈ ₹{pps:,.0f} per sq ft{" per month" if listing == "Rent" else ""}</div>
                {range_meter(listing, result["low"], result["estimate"], result["high"])}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("The marker shows the estimate. The bar spans the likely range (about 80% of similar homes).")

        loc_sub = df[(df.location == prop["location"]) & (df.listing_type == listing)]
        if len(loc_sub):
            med = loc_sub["price"].median()
            diff = ((result["estimate"] / med) - 1) * 100 if med else 0
            direction = "above" if diff > 0.5 else "below" if diff < -0.5 else "in line with"
            tail = f" ({abs(diff):.1f}%)" if direction != "in line with" else ""
            st.markdown(
                f"""
                <div class="note">
                    Typical {noun} in <b>{escape(prop["location"])}</b> is
                    <b>{escape(fmt(listing, med))}</b>. Your home comes out
                    <b>{direction}</b> that{tail}.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        st.caption(
            "A model estimate, not a valuation, listing quote or investment advice. "
            "Always check with a local broker or registered valuer before deciding."
        )


with tab_est:
    estimate_view(listing)


# ============================================================
# COMPARE
# ============================================================
@st.fragment
def compare_view(listing: str) -> None:
    head, action = st.columns([4, 1], vertical_alignment="bottom")
    with head:
        st.markdown('<div class="section-title">Two homes, side by side</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-sub">Change anything on either home and the comparison updates.</p>',
            unsafe_allow_html=True,
        )
    action.button("Swap A and B", key="swap_ab", on_click=_swap_ab, width="stretch")

    ca, cb = st.columns(2, gap="large")
    with ca:
        st.markdown("### Home A")
        pa = property_form("A", A_DEF)
    with cb:
        st.markdown("### Home B")
        pb = property_form("B", B_DEF)

    ra, rb = estimate(listing, pa), estimate(listing, pb)
    pps_a = per_sqft(listing, ra["estimate"], pa["area_sqft"])
    pps_b = per_sqft(listing, rb["estimate"], pb["area_sqft"])
    unit = " / month" if listing == "Rent" else ""

    st.divider()
    st.markdown('<div class="section-title">The comparison</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        stat_card(f"Home A · {pa['location']}", fmt(listing, ra["estimate"]), f"₹{pps_a:,.0f} per sq ft{unit}")
    with m2:
        delta = ((rb["estimate"] / ra["estimate"]) - 1) * 100 if ra["estimate"] else 0
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "="
        stat_card(
            f"Home B · {pb['location']}", fmt(listing, rb["estimate"]),
            f"{arrow} {abs(delta):.1f}% {'more' if delta > 0 else 'less' if delta < 0 else 'same'} than Home A",
        )
    with m3:
        gap = abs(pps_a - pps_b)
        if gap < 1:
            stat_card("Price per sq ft", "About equal", "Neither home is cheaper per sq ft", good=True)
        else:
            cheaper = "A" if pps_a < pps_b else "B"
            pct = gap / max(pps_a, pps_b) * 100
            stat_card(
                "Lower price per sq ft", f"Home {cheaper}", f"₹{gap:,.0f} less ({pct:.0f}% lower)", good=True,
            )

    st.markdown(
        '<div class="note" style="margin-top:1rem;">Price per sq ft is one lens. Floor, age, '
        "amenities and the walk to the metro can matter just as much, so use this alongside a site visit.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Side-by-side details")
    table = pd.DataFrame(
        {
            "": [
                "Neighbourhood", "Type", "BHK / bathrooms", "Area", "Floor", "Age (years)",
                "Furnishing", "Metro distance", "Estimate", "Likely range", "₹ per sq ft",
            ],
            "Home A": [
                pa["location"], pa["property_type"], f"{pa['bhk']} / {pa['bathrooms']}",
                f"{pa['area_sqft']:,.0f} sq ft", f"{pa['floor']} of {pa['total_floors']}",
                f"{pa['age_years']:.1f}", pa["furnishing"], f"{pa['metro_dist_km']:.1f} km",
                fmt(listing, ra["estimate"]), f"{fmt(listing, ra['low'])} – {fmt(listing, ra['high'])}",
                f"{pps_a:,.0f}",
            ],
            "Home B": [
                pb["location"], pb["property_type"], f"{pb['bhk']} / {pb['bathrooms']}",
                f"{pb['area_sqft']:,.0f} sq ft", f"{pb['floor']} of {pb['total_floors']}",
                f"{pb['age_years']:.1f}", pb["furnishing"], f"{pb['metro_dist_km']:.1f} km",
                fmt(listing, rb["estimate"]), f"{fmt(listing, rb['low'])} – {fmt(listing, rb['high'])}",
                f"{pps_b:,.0f}",
            ],
        }
    ).astype(str)

    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button(
        "Download comparison (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="propcompare-comparison.csv",
        mime="text/csv",
    )

    chart_df = pd.DataFrame({"Home": ["A", "B"], "pps": [pps_a, pps_b]})
    base = alt.Chart(chart_df).encode(
        x=alt.X("Home:N", axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y("pps:Q", title=f"₹ per sq ft{unit}"),
    )
    bars = base.mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, size=90).encode(
        color=alt.Color("Home:N", legend=None, scale=alt.Scale(domain=["A", "B"], range=[SEA, MARIGOLD])),
        tooltip=[alt.Tooltip("Home:N"), alt.Tooltip("pps:Q", title="₹ per sq ft", format=",.0f")],
    )
    labels = base.mark_text(dy=-8, color=INK, fontWeight=600).encode(text=alt.Text("pps:Q", format=",.0f"))
    show_chart((bars + labels).properties(height=300))


with tab_cmp:
    compare_view(listing)


# ============================================================
# MARKET
# ============================================================
@st.fragment
def market_view(listing: str) -> None:
    st.markdown('<div class="section-title">How neighbourhoods compare</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Typical prices across the neighbourhoods in the demo dataset. '
        "Pick one to see where it sits.</p>",
        unsafe_allow_html=True,
    )

    divisor, price_title = axis_scale(listing)
    sub = df[df.listing_type == listing].copy()
    sub["price_scaled"] = sub["price"] / divisor

    highlight = st.selectbox("Highlight a neighbourhood", ["None"] + LOCS, key="mkt_highlight")

    by_loc = (
        sub.groupby(["location", "city"], as_index=False)["price_scaled"]
        .median()
        .sort_values("price_scaled", ascending=False)
    )
    by_loc["Typical"] = [fmt(listing, v * divisor) for v in by_loc["price_scaled"]]

    bars = (
        alt.Chart(by_loc)
        .mark_bar(cornerRadiusEnd=7)
        .encode(
            y=alt.Y("location:N", sort="-x", title=None),
            x=alt.X("price_scaled:Q", title=f"Typical {price_title.lower()}"),
            color=alt.condition(
                alt.datum.location == highlight, alt.value(MARIGOLD), alt.value(SEA)
            ),
            tooltip=[
                alt.Tooltip("location:N", title="Neighbourhood"),
                alt.Tooltip("city:N", title="City"),
                alt.Tooltip("Typical:N", title="Typical"),
            ],
        )
        .properties(height=max(420, len(by_loc) * 30))
    )
    show_chart(bars)

    # Sampling keeps the scatter snappy on large datasets.
    plot_df = sub.sample(min(len(sub), 1500), random_state=7) if len(sub) else sub
    bhk_domain = sorted(plot_df["bhk"].unique().tolist()) if len(plot_df) else []

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### Size and price")
        scatter = (
            alt.Chart(plot_df)
            .mark_circle(opacity=0.6, size=55)
            .encode(
                x=alt.X("area_sqft:Q", title="Area (sq ft)"),
                y=alt.Y("price_scaled:Q", title=price_title),
                color=alt.Color(
                    "bhk:O", title="BHK",
                    scale=alt.Scale(domain=bhk_domain, range=CHART_RANGE[: len(bhk_domain)]),
                ),
                tooltip=[
                    alt.Tooltip("location:N", title="Neighbourhood"),
                    alt.Tooltip("bhk:O", title="BHK"),
                    alt.Tooltip("area_sqft:Q", title="Area (sq ft)", format=",.0f"),
                    alt.Tooltip("price_scaled:Q", title=price_title, format=",.2f"),
                ],
            )
            .interactive()
            .properties(height=360)
        )
        show_chart(scatter)
        st.caption("Scroll or pinch to zoom, drag to pan.")

    with c2:
        st.markdown("### Price range by bedrooms")
        box = (
            alt.Chart(plot_df)
            .mark_boxplot(extent="min-max", color=SEA, median=dict(color=INK))
            .encode(
                x=alt.X("bhk:O", title="BHK", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("price_scaled:Q", title=price_title),
            )
            .properties(height=360)
        )
        show_chart(box)

    st.markdown("### The dataset at a glance")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Neighbourhoods", sub["location"].nunique())
    s2.metric("Homes", f"{len(sub):,}")
    s3.metric("Median area", f"{sub['area_sqft'].median():,.0f} sq ft")
    s4.metric("Median BHK", f"{sub['bhk'].median():.0f}")


with tab_mkt:
    market_view(listing)


# ============================================================
# MODEL
# ============================================================
def model_view() -> None:
    st.markdown('<div class="section-title">How the estimate is made</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="note">
            There is one model for sales and one for rents. Each learns from past listings
            how size, location, age, floor and amenities move a price. Neighbourhood averages
            are calculated without peeking at the homes used for testing, so the scores below
            are an honest check rather than a flattering one.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    for lt in ("Sale", "Rent"):
        b = models[lt]
        st.markdown(f"### {'Sale' if lt == 'Sale' else 'Rent'} model")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fit (R²)", f"{b.metrics['R2']:.3f}", help="1.0 means a perfect match on unseen homes.")
        c2.metric("Typical error (MAPE)", f"{b.metrics['MAPE_%']:.1f}%", help="Average gap between estimate and actual price.")
        c3.metric("Average miss (MAE)", fmt(lt, b.metrics["MAE"]))
        c4.metric("Train / test homes", f"{b.metrics['train_rows']} / {b.metrics['test_rows']}")

        imp = b.importances.head(10).reset_index()
        imp.columns = ["feature", "importance"]
        imp["feature"] = imp["feature"].astype(str).str.replace("_", " ", regex=False).str.capitalize()

        st.markdown("**What matters most to the price**")
        chart = (
            alt.Chart(imp)
            .mark_bar(cornerRadiusEnd=5, color=SEA)
            .encode(
                y=alt.Y("feature:N", sort="-x", title=None),
                x=alt.X("importance:Q", title="Relative importance"),
                tooltip=[alt.Tooltip("feature:N", title="Factor"), alt.Tooltip("importance:Q", format=".3f")],
            )
            .properties(height=300)
        )
        show_chart(chart)

    with st.expander("Preview the dataset"):
        st.dataframe(df.head(100), width="stretch")

    with st.expander("What these scores don't tell you", expanded=True):
        st.write(
            "The listings here are generated, not collected from the market. The scores show how well "
            "the model learned that generated data, not how accurate it would be on real Mumbai prices. "
            "To use it for real decisions, replace the generator with verified or licensed listing data "
            "and retrain."
        )


with tab_model:
    model_view()


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        PropCompare · Mumbai &amp; Navi Mumbai<br>
        Built for exploring, not for financial or real estate advice.<br>
        <a href="#top">Back to top</a>
    </div>
    """,
    unsafe_allow_html=True,
)
