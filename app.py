"""PropCompare — Mumbai & Navi Mumbai property estimates.

Run with:  streamlit run app.py
Needs Streamlit >= 1.50 (fragments, st.container(key=...), width="stretch").

Real photos (optional): drop small images (<= ~200 KB each) into
assets/properties/ named after a neighbourhood or property type, e.g.
andheri-west.jpg, kharghar.webp, villa.jpg. When a match exists it replaces
the built-in illustration for that home.
"""

from __future__ import annotations

import base64
import re
from html import escape
from pathlib import Path

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

INK = "#12262B"
MUTED = "#4B6168"
SEA = "#17646E"
MARIGOLD = "#E39B2B"
GRID = "#DCE5E3"
CHART_RANGE = ["#17646E", "#E39B2B", "#7A4E8C", "#3E8E5A", "#C4573A", "#4C7BD9"]
ASSET_DIR = Path(__file__).parent / "assets" / "properties"

alt.data_transformers.disable_max_rows()


def H(markup: str) -> str:
    """Flatten HTML to one line.

    Markdown treats indented or blank-separated lines as code blocks, which can
    break nested HTML. Collapsing it avoids that entirely.
    """
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def html(markup: str) -> None:
    st.markdown(H(markup), unsafe_allow_html=True)


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

    /* Minimal skeuomorphism: a lit top edge, a faint bottom edge, a whisper of lift */
    --raise: inset 0 1px 0 rgba(255,255,255,.95), inset 0 -1px 0 rgba(18,38,43,.05),
             0 1px 2px rgba(18,38,43,.07), 0 12px 18px -16px rgba(18,38,43,.35);
    --press: inset 0 1px 3px rgba(18,38,43,.14), inset 0 0 0 1px rgba(18,38,43,.03);
    --grain: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 .07 0 0 0 0 .15 0 0 0 0 .17 .3 0 0 0 -.1'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
}

/* ---------- Base & smooth scrolling ---------- */
html, [data-testid="stMain"], section.main, .main {
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
.block-container { max-width: 1280px; padding: 1rem 2.5rem 4rem; }

h1, h2, h3, h4, .section-title, .hero h1 {
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
    position: absolute; left: -9999px; top: .5rem;
    background: var(--ink); color: #fff !important;
    padding: .6rem 1rem; border-radius: 10px; z-index: 1000;
}
.skip-link:focus { left: .75rem; }

/* ---------- Sticky top bar ---------- */
.st-key-topbar {
    position: sticky; top: 0; z-index: 60;
    margin: 0 -1rem 1rem; padding: .55rem 1rem;
    background: rgba(238,243,241,.82);
    -webkit-backdrop-filter: blur(14px) saturate(1.2);
    backdrop-filter: blur(14px) saturate(1.2);
    border-bottom: 1px solid var(--line);
}
.brand {
    display: flex; align-items: center; gap: .65rem;
    font-family: "Bricolage Grotesque", sans-serif;
    font-weight: 800; font-size: 1.2rem; letter-spacing: -.02em;
}
.brand-mark {
    width: 30px; height: 30px; border-radius: 9px;
    background: linear-gradient(145deg, #1d7580, var(--sea-deep));
    box-shadow: inset 0 1px 0 rgba(255,255,255,.3), 0 1px 2px rgba(18,38,43,.3);
    display: grid; place-items: center;
}
.brand-mark svg { width: 18px; height: 18px; }
.brand small { font-family: "Figtree", sans-serif; font-weight: 500; font-size: .85rem; color: var(--muted); }

/* Market toggle: an inset track with a raised knob */
div[role="radiogroup"] {
    display: inline-flex; gap: 2px; padding: 4px; margin-left: auto;
    border-radius: 999px; background: #e1e9e7; box-shadow: var(--press);
}
div[role="radiogroup"] label {
    border: 0; background: transparent; border-radius: 999px; padding: .28rem 1rem;
    transition: background .25s var(--ease), box-shadow .25s var(--ease);
}
div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(#fff, #f1f6f5);
    box-shadow: 0 1px 2px rgba(18,38,43,.22), inset 0 1px 0 #fff;
}
div[role="radiogroup"] label:has(input:checked) p { font-weight: 700; }

/* ---------- Hero ---------- */
.hero {
    display: grid; grid-template-columns: 1.35fr .85fr; align-items: center; gap: 1rem;
    padding: 2.4rem 2.6rem; border-radius: 30px;
    background: linear-gradient(120deg, rgba(255,255,255,.94), rgba(255,255,255,.66)), var(--surface);
    border: 1px solid var(--line);
    box-shadow: var(--raise);
    overflow: hidden;
}
.hero-copy > * { animation: rise .8s var(--ease) both; }
.hero-copy > *:nth-child(2) { animation-delay: .08s; }
.hero-copy > *:nth-child(3) { animation-delay: .16s; }
.hero h1 { font-size: clamp(2rem, 4.2vw, 3.5rem); line-height: 1.04; font-weight: 800; margin: 0 0 .9rem; }
.hero p { max-width: 52ch; color: var(--muted); font-size: 1.08rem; line-height: 1.65; margin: 0 0 1.2rem; }
.data-note {
    display: inline-flex; gap: .5rem; padding: .5rem .9rem; border-radius: 999px;
    background: var(--marigold-soft); color: #6b4a0b; font-size: .88rem; font-weight: 500;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.7), 0 1px 1px rgba(107,74,11,.15);
}
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }

.scene-wrap { position: relative; height: 340px; perspective: 1400px; }
.sun {
    position: absolute; right: 12%; top: 4%; width: 110px; height: 110px; border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #f6c46a, var(--marigold)); opacity: .9;
}
.scene {
    position: absolute; left: 50%; top: 60%; width: 192px; height: 192px; margin: -96px 0 0 -96px;
    transform-style: preserve-3d; transform: rotateX(58deg) rotateZ(42deg);
    animation: sway 16s ease-in-out 2 alternate;
}
@keyframes sway {
    from { transform: rotateX(58deg) rotateZ(38deg); }
    to   { transform: rotateX(58deg) rotateZ(48deg); }
}
.ground {
    position: absolute; inset: -34px; border-radius: 16px; transform: translateZ(-1px);
    background:
        linear-gradient(rgba(23,100,110,.12) 1px, transparent 1px) 0 0 / 24px 24px,
        linear-gradient(90deg, rgba(23,100,110,.12) 1px, transparent 1px) 0 0 / 24px 24px,
        var(--sea-soft);
}
.bld { position: absolute; left: var(--x); top: var(--y); width: 52px; height: 52px; transform-style: preserve-3d; }
.bld i { position: absolute; display: block; }
.bld .top { inset: 0; transform: translateZ(var(--h)); background: var(--c-top); }
.bld .fy {
    left: 0; top: 100%; width: 100%; height: var(--h);
    transform-origin: top; transform: rotateX(90deg); background-color: var(--c-front);
    background-image:
        repeating-linear-gradient(90deg, transparent 0 8px, rgba(255,255,255,.26) 8px 12px),
        repeating-linear-gradient(0deg, transparent 0 10px, rgba(7,30,36,.16) 10px 11px);
}
.bld .fx {
    left: 100%; top: 0; width: var(--h); height: 100%;
    transform-origin: left; transform: rotateY(-90deg); background-color: var(--c-side);
    background-image:
        repeating-linear-gradient(0deg, transparent 0 8px, rgba(255,255,255,.2) 8px 12px),
        repeating-linear-gradient(90deg, transparent 0 10px, rgba(7,30,36,.18) 10px 11px);
}
.t-a { --c-top: #d8ebea; --c-front: #2f8a93; --c-side: #1b5f68; }
.t-b { --c-top: #fbebcb; --c-front: #e8b357; --c-side: #c58a2a; }
.t-c { --c-top: #e6eceb; --c-front: #7fa5ab; --c-side: #5b8189; }

/* ---------- Headings ---------- */
.section-title { font-size: 1.6rem; font-weight: 700; margin: .4rem 0 .15rem; }
.section-sub { color: var(--muted); margin: 0 0 1rem; max-width: 62ch; }
.group-title {
    font-weight: 700; font-size: 1rem; margin: 1.3rem 0 .3rem;
    padding-bottom: .35rem; border-bottom: 1px solid var(--line);
}

/* ---------- Bento grid ---------- */
.bento {
    display: grid;
    grid-template-columns: repeat(var(--cols, 2), minmax(0, 1fr));
    gap: .8rem;
    margin-bottom: .8rem;
}
.cols-1 { --cols: 1; }
.cols-2 { --cols: 2; }
.cols-4 { --cols: 4; }
.span-2 { grid-column: span 2; }
.span-4 { grid-column: span 4; }

.tile {
    position: relative;
    min-width: 0;
    padding: 1rem 1.1rem;
    border: 1px solid var(--line);
    border-radius: 20px;
    background: linear-gradient(180deg, #ffffff, #f6f9f8);
    box-shadow: var(--raise);
}
.tile .k { color: var(--muted); font-size: .88rem; font-weight: 500; }
.tile .v {
    font-family: "Bricolage Grotesque", sans-serif;
    font-weight: 700; font-size: 1.5rem; letter-spacing: -.02em; margin-top: .15rem;
    text-shadow: 0 1px 0 #fff;
    overflow-wrap: anywhere;
}
.tile .v.sm { font-size: 1.05rem; }
.tile .s { color: var(--muted); font-size: .88rem; margin-top: .2rem; }
.tile.good { background: linear-gradient(180deg, #f4fbfa, #e8f4f2); border-color: #a9cfcb; }

/* Ink tile: the estimate, like a dark inlaid plate */
.tile-ink {
    border: 1px solid #0b2a30;
    background: linear-gradient(165deg, #17595f, var(--sea-deep));
    color: #f2f8f7;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.18), inset 0 -2px 6px rgba(0,0,0,.25),
                0 14px 22px -18px rgba(14,50,56,.8);
    padding: 1.3rem 1.4rem;
}
.tile-ink .k, .tile-ink .s { color: #bad7d5; }
.tile-ink .v {
    font-size: clamp(2rem, 3.2vw, 2.8rem); font-weight: 800; letter-spacing: -.03em;
    text-shadow: 0 -1px 0 rgba(0,0,0,.35);
}

/* Range meter: engraved groove + glossy knob */
.meter { margin-top: 1.1rem; }
.meter-track {
    position: relative; height: 9px; border-radius: 99px;
    background: linear-gradient(90deg, rgba(0,0,0,.28), rgba(227,155,43,.55));
    box-shadow: inset 0 1px 3px rgba(0,0,0,.45), 0 1px 0 rgba(255,255,255,.12);
}
.meter-dot {
    position: absolute; top: 50%; width: 22px; height: 22px; margin: -11px 0 0 -11px; border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, #ffd88b, var(--marigold) 60%, #c47f12);
    border: 2px solid #fff; box-shadow: 0 2px 5px rgba(0,0,0,.35), inset 0 1px 1px rgba(255,255,255,.6);
}
.meter-ends { display: flex; justify-content: space-between; margin-top: .6rem; font-size: .85rem; color: #d3e6e4; }

/* Photo tile with layered, parallax illustration */
.tile-photo { padding: 0; overflow: hidden; aspect-ratio: 16 / 9; margin: 0; }
.photo-media { position: absolute; inset: 0; }
.photo-media svg, .photo-media img { width: 100%; height: 100%; object-fit: cover; display: block; }
.photo-media .p-far, .photo-media .p-mid, .photo-media .p-near { transition: transform .7s var(--ease); }
.tile-photo:hover .p-far  { transform: translateX(-8px); }
.tile-photo:hover .p-mid  { transform: translateX(-16px); }
.tile-photo:hover .p-near { transform: translateX(-5px); }
.glass {
    position: absolute; left: .7rem; right: .7rem; bottom: .7rem;
    display: flex; flex-direction: column; gap: .05rem;
    padding: .6rem .85rem; border-radius: 14px; font-size: .85rem; color: var(--ink);
    background: rgba(255,255,255,.9);
    border: 1px solid rgba(255,255,255,.7);
    box-shadow: inset 0 1px 0 #fff, 0 6px 12px -10px rgba(18,38,43,.5);
}
.glass b { font-family: "Bricolage Grotesque", sans-serif; font-size: 1.05rem; }
.glass span { color: var(--muted); }
.badge-ill {
    position: absolute; top: .6rem; right: .6rem; padding: .15rem .6rem; border-radius: 999px;
    font-size: .75rem; font-weight: 600; color: var(--ink); background: rgba(255,255,255,.8);
}

/* Amenity chips: small pressed/raised pills */
.chips { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .55rem; }
.chip {
    padding: .22rem .65rem; border-radius: 999px; font-size: .85rem; font-weight: 600;
    background: linear-gradient(#fff, #eef4f3); color: var(--ink); box-shadow: var(--raise);
}
.chip.off { background: #e9efed; color: #5c6f75; font-weight: 500; box-shadow: var(--press); }

.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(90px, 1fr)); gap: .6rem 1rem; margin-top: .6rem; }
.facts div { display: flex; flex-direction: column; }
.facts span { color: var(--muted); font-size: .82rem; }
.facts b { font-size: 1rem; }

.note {
    background: linear-gradient(#fff, #f8fbfa);
    border: 1px solid var(--line); border-left: 4px solid var(--marigold);
    border-radius: 14px; padding: .85rem 1.1rem; line-height: 1.6; box-shadow: var(--raise);
}

/* 3D tilt (hover only) */
.tilt3d { transition: transform .45s var(--ease); transform: perspective(1000px); }
.tilt3d:hover { transform: perspective(1000px) rotateX(2deg) rotateY(-3deg) translateY(-3px); }

.jump { display: none; }

/* ---------- Streamlit widgets (inset fields, raised buttons) ---------- */
[data-testid="stWidgetLabel"] p { font-weight: 600; font-size: .9rem; color: var(--ink); }
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] > div {
    background: #fff !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    box-shadow: var(--press);
}
div.stButton > button, div.stDownloadButton > button {
    border-radius: 999px; border: 1px solid var(--line); color: var(--ink);
    font-weight: 600; padding: .4rem 1.1rem;
    background: linear-gradient(#fff, #eef4f3); box-shadow: var(--raise);
    transition: transform .2s var(--ease), border-color .2s var(--ease);
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    border-color: var(--sea); color: var(--sea); transform: translateY(-1px);
}
div.stButton > button:active, div.stDownloadButton > button:active {
    transform: translateY(1px); box-shadow: var(--press);
}
button[data-baseweb="tab"] { font-weight: 600; font-size: 1rem; color: var(--muted); padding: .7rem 1rem; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--sea); }
div[data-baseweb="tab-highlight"] { background: var(--marigold) !important; height: 3px !important; }
div[data-baseweb="tab-border"] { background: var(--line) !important; }
div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
[data-testid="stMetric"] {
    background: linear-gradient(#fff, #f6f9f8); border: 1px solid var(--line);
    border-radius: 16px; padding: .8rem 1rem; box-shadow: var(--raise);
}
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 14px; background: #fff; box-shadow: var(--raise); }

.footer { text-align: center; color: var(--muted); font-size: .88rem; line-height: 1.7; padding: 2.5rem 0 .5rem; }
.footer a { color: var(--sea); font-weight: 600; text-decoration: none; }
.footer a:hover { text-decoration: underline; }

#MainMenu { visibility: hidden; }
[data-testid="stFooter"], .stApp > footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- Responsive ---------- */
@media (max-width: 900px) {
    .block-container { padding: .5rem 1rem 3rem; }
    .hero { grid-template-columns: 1fr; padding: 1.6rem 1.4rem; border-radius: 22px; }
    .scene-wrap { display: none; }
    .cols-4 { --cols: 2; }
    .span-4 { grid-column: span 2; }
    .jump {
        display: inline-block; margin: .4rem 0 .2rem; padding: .45rem .95rem; border-radius: 999px;
        background: var(--sea); color: #fff !important; font-weight: 600; text-decoration: none;
    }
    .st-key-topbar { margin: 0 -.5rem .75rem; }
}
@media (max-width: 520px) {
    .bento { --cols: 1; }
    .span-2, .span-4 { grid-column: auto; }
}

/* ---------- Motion preferences ---------- */
@media (prefers-reduced-motion: reduce) {
    html, [data-testid="stMain"], section.main, .main { scroll-behavior: auto; }
    *, *::before, *::after { animation: none !important; transition-duration: .01ms !important; }
    .tilt3d:hover { transform: none; }
    .tile-photo:hover .p-far, .tile-photo:hover .p-mid, .tile-photo:hover .p-near { transform: none; }
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


df = None
models = None

LOCS = sorted(LOCATIONS)


@st.cache_data(show_spinner=False, max_entries=512)
def _predict_cached(listing: str, prop_items: tuple, _models) -> dict:
    return predict(_models, listing, dict(prop_items))


def estimate(listing: str, prop: dict) -> dict:
    return _predict_cached(listing, tuple(sorted(prop.items())), models)


# ============================================================
# FORM STATE, PRESETS
# ============================================================
FORM_SUFFIXES = [
    "loc", "pt", "bhk", "bath", "area", "tf", "fl", "age",
    "fu", "pk", "mt", "sc", "lift", "gym", "pool", "sec",
]

A_DEF = dict(
    location="Andheri West", property_type="Apartment", bhk=2, bathrooms=2, area_sqft=850,
    floor=8, total_floors=20, age_years=4.0, furn_idx=1, parking=1, metro_dist_km=0.8,
    school_dist_km=1.0, lift=True, gym=True, pool=False, security=True,
)
B_DEF = dict(
    A_DEF, location="Kharghar", area_sqft=950, floor=10, total_floors=22,
    age_years=2.0, metro_dist_km=1.5, pool=True,
)
COMPACT_DEF = dict(
    A_DEF, bhk=1, bathrooms=1, area_sqft=480, floor=4, total_floors=12,
    age_years=8.0, furn_idx=0, parking=0, pool=False, gym=False,
)
PRESETS = {
    "2 BHK, Andheri West": A_DEF,
    "Roomy 2 BHK, Kharghar": B_DEF,
    "Compact 1 BHK": COMPACT_DEF,
}


def _furn_value(idx: int) -> str:
    return FURNISHING[max(0, min(int(idx), len(FURNISHING) - 1))]


def _apply_preset(key: str, d: dict) -> None:
    values = {
        "loc": d["location"] if d["location"] in LOCS else LOCS[0],
        "pt": d["property_type"], "bhk": int(d["bhk"]), "bath": int(d["bathrooms"]),
        "area": int(d["area_sqft"]), "tf": int(d["total_floors"]), "fl": int(d["floor"]),
        "age": float(d["age_years"]), "fu": _furn_value(d["furn_idx"]), "pk": int(d["parking"]),
        "mt": float(d["metro_dist_km"]), "sc": float(d["school_dist_km"]),
        "lift": bool(d["lift"]), "gym": bool(d["gym"]), "pool": bool(d["pool"]), "sec": bool(d["security"]),
    }
    for suffix, value in values.items():
        st.session_state[f"{key}_{suffix}"] = value


def _swap_ab() -> None:
    for s in FORM_SUFFIXES:
        a, b = st.session_state.get(f"A_{s}"), st.session_state.get(f"B_{s}")
        if a is not None and b is not None:
            st.session_state[f"A_{s}"], st.session_state[f"B_{s}"] = b, a


# ============================================================
# PROPERTY IMAGERY (built-in illustrations, optional real photos)
# ============================================================
PALETTES = [  # sky top, sky bottom, far, mid, wall, ground
    ("#cfe6e8", "#f7e6c4", "#b7d3d6", "#8fb5ba", "#e9f1f0", "#c9dcd2"),
    ("#f6dfc0", "#fbf1dc", "#e3c9a4", "#c9a57a", "#f7efe3", "#d8d2b6"),
    ("#d6dff2", "#eef0f8", "#b8c4e0", "#93a4cc", "#eef1f8", "#cdd8cd"),
]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@st.cache_data(show_spinner=False)
def _photo_uri(location: str, ptype: str) -> str | None:
    """Return a data URI for assets/properties/<neighbourhood or type>.<ext>, if present."""
    mimes = {".webp": "image/webp", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    for stem in (_slug(location), _slug(ptype)):
        for ext, mime in mimes.items():
            path = ASSET_DIR / f"{stem}{ext}"
            if path.exists():
                return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    return None


def scene_svg(prop: dict, uid: str) -> tuple[str, str]:
    """A small illustrated scene that reflects the inputs: storeys, your floor, pool, parking, metro."""
    pal = PALETTES[sum(map(ord, prop["location"])) % len(PALETTES)]
    sky_t, sky_b, far, mid, wall, ground = pal
    ptype = prop["property_type"].lower()
    house = any(k in ptype for k in ("villa", "bungalow", "house", "row"))
    total, floor = max(1, int(prop["total_floors"])), int(prop["floor"])
    parts: list[str] = []

    # Skyline layers (parallax on hover)
    far_blocks = [(10, 150, 46), (62, 120, 40), (110, 160, 36), (330, 130, 48), (390, 100, 42), (435, 150, 38)]
    mid_blocks = [(0, 175, 60), (66, 150, 44), (352, 165, 52), (410, 140, 56)]
    parts.append('<g class="p-far">' + "".join(
        f'<rect x="{x}" y="{y}" width="{w}" height="{270 - y}" fill="{far}"/>' for x, y, w in far_blocks) + "</g>")
    parts.append('<g class="p-mid">' + "".join(
        f'<rect x="{x}" y="{y}" width="{w}" height="{270 - y}" fill="{mid}"/>' for x, y, w in mid_blocks) + "</g>")

    near: list[str] = []
    if house:
        win_n = min(max(int(prop["bhk"]), 2), 4)
        near.append(f'<rect x="150" y="170" width="190" height="94" rx="3" fill="{wall}"/>')
        near.append('<polygon points="136,174 245,106 354,174" fill="#b5654a"/>')
        near.append('<rect x="300" y="120" width="18" height="40" fill="#9c5540"/>')
        near.append('<rect x="228" y="212" width="34" height="52" rx="3" fill="#5a3b2e"/>')
        span = 160 / win_n
        for i in range(win_n):
            wx = 160 + i * span + 6
            if 205 < wx + 10 < 270:
                continue
            near.append(f'<rect x="{wx:.0f}" y="190" width="22" height="24" rx="2" fill="#9ec8cf"/>')
        label = f"Illustration of a {ptype} with {win_n} windows"
    else:
        n = min(max(total, 2), 26)
        fh = max(6.0, min(15.0, 200 / n))
        tower_h = n * fh
        bx, bw, by = 170, 140, 264 - tower_h
        near.append(f'<rect x="{bx}" y="{by:.1f}" width="{bw}" height="{tower_h:.1f}" rx="3" fill="{wall}"/>')
        near.append(f'<rect x="{bx - 4}" y="{by - 6:.1f}" width="{bw + 8}" height="7" rx="2" fill="{mid}"/>')
        hi = min(n - 1, round(floor / total * (n - 1))) if total > 1 else 0
        for i in range(n):
            level = n - 1 - i
            wy = by + i * fh + fh * 0.2
            for c in range(4):
                wx = bx + 10 + c * 34
                lit = level == hi and c in (1, 2)
                fill = MARIGOLD if lit else ("#8fbcc4" if (i * 3 + c) % 5 else "#b9d8dd")
                near.append(f'<rect x="{wx}" y="{wy:.1f}" width="18" height="{fh * 0.55:.1f}" rx="1.5" fill="{fill}"/>')
        label = f"Illustration of a {total}-storey {ptype} building with floor {floor} highlighted"

    # Extras that echo the inputs
    near.append(f'<rect x="0" y="264" width="480" height="36" fill="{ground}"/>')
    near.append('<rect x="0" y="286" width="480" height="14" fill="#9fb0ad" opacity=".55"/>')
    if prop.get("pool"):
        near.append('<rect x="352" y="270" width="96" height="16" rx="6" fill="#6cc3d5"/>'
                    '<path d="M362 278q8-5 16 0t16 0t16 0t16 0" stroke="#fff" stroke-width="1.5" fill="none" opacity=".8"/>')
    for i in range(min(int(prop.get("parking", 0)), 2)):
        cx = 40 + i * 52
        near.append(f'<rect x="{cx}" y="266" width="40" height="13" rx="5" fill="{SEA if i == 0 else MARIGOLD}"/>'
                    f'<circle cx="{cx + 9}" cy="280" r="4" fill="{INK}"/><circle cx="{cx + 31}" cy="280" r="4" fill="{INK}"/>')
    for tx in (126, 328):
        near.append(f'<rect x="{tx - 2}" y="246" width="4" height="20" fill="#6b5240"/>'
                    f'<circle cx="{tx}" cy="240" r="15" fill="#4f9a73"/>')
    if float(prop.get("metro_dist_km", 9)) < 1.0:
        near.append(f'<rect x="446" y="226" width="3" height="40" fill="{INK}"/>'
                    f'<rect x="436" y="210" width="24" height="18" rx="4" fill="{SEA}"/>'
                    '<text x="448" y="223.5" font-size="12" font-weight="700" fill="#fff" text-anchor="middle" font-family="sans-serif">M</text>')
    parts.append('<g class="p-near">' + "".join(near) + "</g>")

    svg = (
        f'<svg viewBox="0 0 480 300" preserveAspectRatio="xMidYMid slice" role="img" aria-label="{escape(label)}" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<defs><linearGradient id="sky{uid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{sky_t}"/><stop offset="1" stop-color="{sky_b}"/></linearGradient></defs>'
        f'<rect width="480" height="300" fill="url(#sky{uid})"/>'
        '<circle cx="396" cy="62" r="30" fill="#f6c46a" opacity=".85"/>'
        + "".join(parts) + "</svg>"
    )
    return svg, label


# ============================================================
# BENTO TILES
# ============================================================
def tile(label: str, value: str, sub: str = "", *, cls: str = "", small: bool = False) -> str:
    v_cls = "v sm" if small else "v"
    return (
        f'<div class="tile {cls}"><div class="k">{escape(label)}</div>'
        f'<div class="{v_cls}">{escape(value)}</div>'
        + (f'<div class="s">{escape(sub)}</div>' if sub else "")
        + "</div>"
    )


def photo_tile(prop: dict, uid: str, cls: str = "") -> str:
    uri = _photo_uri(prop["location"], prop["property_type"])
    if uri:
        alt_text = f"{prop['property_type']} in {prop['location']}"
        media, badge = f'<img src="{uri}" alt="{escape(alt_text)}" loading="lazy">', ""
    else:
        media, _ = scene_svg(prop, uid)
        badge = '<span class="badge-ill">Illustration</span>'
    meta = f"{prop['bhk']} BHK · {prop['area_sqft']:,.0f} sq ft · floor {prop['floor']} of {prop['total_floors']}"
    return (
        f'<figure class="tile tile-photo tilt3d {cls}"><div class="photo-media">{media}</div>{badge}'
        f'<figcaption class="glass"><b>{escape(prop["location"])}</b>'
        f'<span>{escape(prop["city"])} · {escape(prop["property_type"])}</span>'
        f'<span>{escape(meta)}</span></figcaption></figure>'
    )


def range_meter(listing: str, low: float, est: float, high: float) -> str:
    span = high - low
    pos = 50 if span <= 0 else max(0, min(100, (est - low) / span * 100))
    label = f"Estimate {fmt(listing, est)}. Likely range {fmt(listing, low)} to {fmt(listing, high)}."
    return (
        f'<div class="meter" role="img" aria-label="{escape(label)}">'
        f'<div class="meter-track"><span class="meter-dot" style="left:{pos:.1f}%"></span></div>'
        f'<div class="meter-ends"><span>{escape(fmt(listing, low))}</span><span>{escape(fmt(listing, high))}</span></div></div>'
    )


def chip(label: str, on: bool) -> str:
    return f'<span class="chip{"" if on else " off"}">{"✓" if on else "✕"} {escape(label)}</span>'


def facts(items: list[tuple[str, str]]) -> str:
    return '<div class="facts">' + "".join(
        f"<div><span>{escape(k)}</span><b>{escape(v)}</b></div>" for k, v in items
    ) + "</div>"


def bento(tiles: list[str], cols: int = 2) -> None:
    html(f'<div class="bento cols-{cols}">' + "".join(tiles) + "</div>")


# ============================================================
# CHART HELPERS
# ============================================================
def axis_scale(listing: str) -> tuple[float, str]:
    return (1e7, "Price (₹ crore)") if listing == "Sale" else (1e3, "Rent (₹ thousand / month)")


def style_chart(chart):
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


# ============================================================
# PROPERTY FORM
# ============================================================
def property_form(key: str, defaults: dict, *, presets: bool = False) -> dict:
    saved = st.session_state.get(f"_saved_{key}")
    if saved:  # restore values after the user switched sections and back
        defaults = {
            **defaults, **saved,
            "furn_idx": FURNISHING.index(saved["furnishing"]) if saved["furnishing"] in FURNISHING else defaults["furn_idx"],
            "lift": bool(saved["lift"]), "gym": bool(saved["gym"]),
            "pool": bool(saved["pool"]), "security": bool(saved["security"]),
        }
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
    area = c3.number_input("Carpet area (sq ft)", 250, 6000, int(defaults["area_sqft"]), step=25, key=f"{key}_area")

    per_bed = area / max(int(bhk), 1)
    if per_bed < 220:
        st.info(f"{int(bhk)} BHK in {int(area):,} sq ft is unusually tight, so the estimate may be less reliable.")
    elif per_bed > 2500:
        st.info(f"{int(area):,} sq ft for {int(bhk)} BHK is unusually spacious, so the estimate may be less reliable.")

    st.markdown('<div class="group-title">The building</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    total = c1.number_input("Floors in building", 1, 60, defaults["total_floors"], key=f"{key}_tf")
    if st.session_state.get(f"{key}_fl", 0) > int(total):  # keep floor valid if the building got shorter
        st.session_state[f"{key}_fl"] = int(total)
    floor = c2.number_input(
        "Floor", 0, int(total), min(defaults["floor"], int(total)), key=f"{key}_fl", help="0 is ground floor.",
    )
    age = c3.number_input("Age (years)", 0.0, 50.0, float(defaults["age_years"]), step=0.5, key=f"{key}_age")

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
    school = c2.slider("Distance to school (km)", 0.3, 5.0, float(defaults["school_dist_km"]), 0.1, key=f"{key}_sc")

    st.markdown('<div class="group-title">Amenities</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    lift = int(a1.checkbox("Lift", defaults["lift"], key=f"{key}_lift"))
    gym = int(a2.checkbox("Gym", defaults["gym"], key=f"{key}_gym"))
    pool = int(a3.checkbox("Pool", defaults["pool"], key=f"{key}_pool"))
    sec = int(a4.checkbox("24×7 security", defaults["security"], key=f"{key}_sec"))

    prop = dict(
        location=loc, city=LOCATIONS[loc][0], property_type=ptype, bhk=int(bhk), bathrooms=int(bath),
        area_sqft=float(area), floor=int(floor), total_floors=int(total), age_years=float(age),
        furnishing=furn, parking=int(parking), lift=lift, gym=gym, pool=pool, security=sec,
        metro_dist_km=float(metro), school_dist_km=float(school),
    )
    st.session_state[f"_saved_{key}"] = prop
    return prop


# ============================================================
# TOP BAR & HERO
# ============================================================
html('<span id="top"></span><a class="skip-link" href="#main-content">Skip to content</a>')

with st.container(key="topbar"):
    left, right = st.columns([3, 2], vertical_alignment="center")
    left.markdown(
        H("""
        <div class="brand">
            <span class="brand-mark" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M4 21V9l6-4v16M10 21V3l10 5v13M2 21h20"/>
                </svg>
            </span>
            PropCompare <small>Mumbai &amp; Navi Mumbai</small>
        </div>
        """),
        unsafe_allow_html=True,
    )
    listing = right.radio(
        "Show prices for", ["Sale", "Rent"],
        format_func=lambda v: "Buying" if v == "Sale" else "Renting",
        horizontal=True, label_visibility="collapsed", key="listing_mode",
    )

SKYLINE = [
    (0, 0, 190, "a"), (70, 0, 130, "b"), (140, 0, 86, "c"),
    (0, 70, 120, "c"), (70, 70, 160, "a"), (140, 70, 70, "b"),
    (0, 140, 64, "b"), (70, 140, 92, "c"), (140, 140, 44, "a"),
]
skyline = "".join(
    f'<div class="bld t-{tone}" style="--x:{x}px;--y:{y}px;--h:{h}px">'
    '<i class="top"></i><i class="fy"></i><i class="fx"></i></div>'
    for x, y, h, tone in SKYLINE
)

html(f"""
<section class="hero" aria-labelledby="hero-title">
    <div class="hero-copy">
        <h1 id="hero-title">Know what a Mumbai home is worth before you talk price.</h1>
        <p>Get a quick estimate for buying or renting, set two homes side by side, and see how
        neighbourhoods compare. Every number comes with a likely range, so you know how much to trust it.</p>
        <span class="data-note" role="note"><b>Demo data.</b> Built on simulated listings, so treat results as a guide.</span>
    </div>
    <div class="scene-wrap" aria-hidden="true">
        <div class="sun"></div>
        <div class="scene"><div class="ground"></div>{skyline}</div>
    </div>
</section>
<div id="main-content" tabindex="-1"></div>
""")

st.write("")

SECTIONS = ["Estimate", "Compare two homes", "Neighbourhoods", "How it works"]
section = st.segmented_control(
    "Section", SECTIONS, default=SECTIONS[0], key="section", label_visibility="collapsed",
) or SECTIONS[0]


# ============================================================
# ESTIMATE
# ============================================================
@st.fragment
def estimate_view(listing: str) -> None:
    form_col, result_col = st.columns([1.3, 1], gap="large")

    with form_col:
        st.markdown('<div class="section-title">Describe the home</div>', unsafe_allow_html=True)
        html('<p class="section-sub">Fill in what you know. The estimate updates as you go.</p>'
             '<a class="jump" href="#estimate-result">Jump to the estimate</a>')
        prop = property_form("single", A_DEF, presets=True)

    result = estimate(listing, prop)
    pps = per_sqft(listing, result["estimate"], prop["area_sqft"])
    noun = "sale price" if listing == "Sale" else "monthly rent"
    unit = " per month" if listing == "Rent" else ""

    loc_sub = df[(df.location == prop["location"]) & (df.listing_type == listing)]
    if len(loc_sub):
        med = loc_sub["price"].median()
        diff = ((result["estimate"] / med) - 1) * 100 if med else 0
        local_val = fmt(listing, med)
        local_sub = (
            "In line with this home" if abs(diff) <= 0.5
            else f"This home is {abs(diff):.1f}% {'above' if diff > 0 else 'below'}"
        )
    else:
        local_val, local_sub = "Not enough data", ""

    around = facts([
        ("Metro", "%.1f km" % prop["metro_dist_km"]),
        ("School", "%.1f km" % prop["school_dist_km"]),
    ])
    building = facts([
        ("Floor", "%d of %d" % (prop["floor"], prop["total_floors"])),
        ("Age", "%.1f yrs" % prop["age_years"]),
        ("Furnishing", str(prop["furnishing"])),
        ("Parking", str(prop["parking"])),
    ])

    with result_col:
        html('<div id="estimate-result"></div>')
        bento(
            [
                photo_tile(prop, "e", "span-2"),
                f'<div class="tile tile-ink tilt3d span-2" role="status" aria-live="polite">'
                f'<div class="k">Estimated {noun}</div>'
                f'<div class="v">{escape(fmt(listing, result["estimate"]))}</div>'
                f'<div class="s">≈ ₹{pps:,.0f} per sq ft{unit}</div>'
                f'{range_meter(listing, result["low"], result["estimate"], result["high"])}</div>',
                tile("Price per sq ft", f"₹{pps:,.0f}", f"Per sq ft{unit}"),
                tile(f"Typical in {prop['location']}", local_val, local_sub, small=True),
                f'<div class="tile"><div class="k">Amenities</div><div class="chips">'
                f'{chip("Lift", prop["lift"])}{chip("Gym", prop["gym"])}{chip("Pool", prop["pool"])}'
                f'{chip("Security", prop["security"])}</div></div>',
                f'<div class="tile"><div class="k">Getting around</div>{around}</div>',
                f'<div class="tile span-2"><div class="k">The building</div>{building}</div>',
            ]
        )
        st.caption("The marker is the estimate. The bar spans the likely range (about 80% of similar homes).")
        st.caption(
            "A model estimate, not a valuation, listing quote or investment advice. "
            "Check with a local broker or registered valuer before deciding."
        )




# ============================================================
# COMPARE
# ============================================================
@st.fragment
def compare_view(listing: str) -> None:
    head, action = st.columns([4, 1], vertical_alignment="bottom")
    with head:
        st.markdown('<div class="section-title">Two homes, side by side</div>', unsafe_allow_html=True)
        html('<p class="section-sub">Change anything on either home and the comparison updates.</p>')
    action.button("Swap A and B", key="swap_ab", on_click=_swap_ab, width="stretch")

    ca, cb = st.columns(2, gap="large")
    with ca:
        st.markdown("### Home A")
        slot_a = st.empty()
        pa = property_form("A", A_DEF)
        slot_a.markdown(H(f'<div class="bento cols-1">{photo_tile(pa, "a")}</div>'), unsafe_allow_html=True)
    with cb:
        st.markdown("### Home B")
        slot_b = st.empty()
        pb = property_form("B", B_DEF)
        slot_b.markdown(H(f'<div class="bento cols-1">{photo_tile(pb, "b")}</div>'), unsafe_allow_html=True)

    ra, rb = estimate(listing, pa), estimate(listing, pb)
    pps_a = per_sqft(listing, ra["estimate"], pa["area_sqft"])
    pps_b = per_sqft(listing, rb["estimate"], pb["area_sqft"])
    unit = " / month" if listing == "Rent" else ""

    delta = ((rb["estimate"] / ra["estimate"]) - 1) * 100 if ra["estimate"] else 0
    gap = abs(pps_a - pps_b)
    if gap < 1:
        verdict_v, verdict_s = "About equal per sq ft", "Neither home is cheaper for the space."
    else:
        cheaper = "A" if pps_a < pps_b else "B"
        pct = gap / max(pps_a, pps_b) * 100
        total_cmp = (
            "about the same overall" if abs(delta) < 0.5
            else f"Home B is {abs(delta):.1f}% {'more' if delta > 0 else 'less'} overall"
        )
        verdict_v = f"Home {cheaper} is cheaper per sq ft"
        verdict_s = f"₹{gap:,.0f} less ({pct:.0f}% lower). {total_cmp[0].upper() + total_cmp[1:]}."

    st.divider()
    st.markdown('<div class="section-title">The comparison</div>', unsafe_allow_html=True)
    bento(
        [
            tile(f"Home A · {pa['location']}", fmt(listing, ra["estimate"]), "Estimate"),
            tile(f"Home B · {pb['location']}", fmt(listing, rb["estimate"]), "Estimate"),
            tile("Better value for the space", verdict_v, verdict_s, cls="good span-2", small=True),
            tile("Home A price per sq ft", f"₹{pps_a:,.0f}", unit.strip() or "Total ÷ area"),
            tile("Home B price per sq ft", f"₹{pps_b:,.0f}", unit.strip() or "Total ÷ area"),
            tile("Home A likely range", f"{fmt(listing, ra['low'])} – {fmt(listing, ra['high'])}", small=True),
            tile("Home B likely range", f"{fmt(listing, rb['low'])} – {fmt(listing, rb['high'])}", small=True),
        ],
        cols=4,
    )
    html('<div class="note">Price per sq ft is one lens. Floor, age, amenities and the walk to the metro can '
         "matter just as much, so use this alongside a site visit.</div>")

    st.markdown("### Side-by-side details")
    table = pd.DataFrame(
        {
            "": ["Neighbourhood", "Type", "BHK / bathrooms", "Area", "Floor", "Age (years)",
                 "Furnishing", "Metro distance", "Estimate", "Likely range", "₹ per sq ft"],
            "Home A": [
                pa["location"], pa["property_type"], f"{pa['bhk']} / {pa['bathrooms']}",
                f"{pa['area_sqft']:,.0f} sq ft", f"{pa['floor']} of {pa['total_floors']}",
                f"{pa['age_years']:.1f}", pa["furnishing"], f"{pa['metro_dist_km']:.1f} km",
                fmt(listing, ra["estimate"]), f"{fmt(listing, ra['low'])} – {fmt(listing, ra['high'])}", f"{pps_a:,.0f}",
            ],
            "Home B": [
                pb["location"], pb["property_type"], f"{pb['bhk']} / {pb['bathrooms']}",
                f"{pb['area_sqft']:,.0f} sq ft", f"{pb['floor']} of {pb['total_floors']}",
                f"{pb['age_years']:.1f}", pb["furnishing"], f"{pb['metro_dist_km']:.1f} km",
                fmt(listing, rb["estimate"]), f"{fmt(listing, rb['low'])} – {fmt(listing, rb['high'])}", f"{pps_b:,.0f}",
            ],
        }
    ).astype(str)
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button(
        "Download comparison (CSV)", table.to_csv(index=False).encode("utf-8"),
        file_name="propcompare-comparison.csv", mime="text/csv",
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




# ============================================================
# MARKET
# ============================================================
@st.fragment
def market_view(listing: str) -> None:
    st.markdown('<div class="section-title">How neighbourhoods compare</div>', unsafe_allow_html=True)
    html('<p class="section-sub">Typical prices across the neighbourhoods in the demo dataset. '
         "Pick one to see where it sits.</p>")

    divisor, price_title = axis_scale(listing)
    sub = df[df.listing_type == listing].copy()
    sub["price_scaled"] = sub["price"] / divisor

    by_loc = (
        sub.groupby(["location", "city"], as_index=False)["price_scaled"]
        .median().sort_values("price_scaled", ascending=False)
    )
    by_loc["Typical"] = [fmt(listing, v * divisor) for v in by_loc["price_scaled"]]

    top, bottom = by_loc.iloc[0], by_loc.iloc[-1]
    bento(
        [
            tile("Priciest neighbourhood", str(top["location"]), f"Typical {top['Typical']}", small=True),
            tile("Most affordable", str(bottom["location"]), f"Typical {bottom['Typical']}", small=True),
            tile("Homes in the dataset", f"{len(sub):,}", f"Across {sub['location'].nunique()} neighbourhoods"),
            tile("Median home", f"{sub['bhk'].median():.0f} BHK", f"{sub['area_sqft'].median():,.0f} sq ft"),
        ],
        cols=4,
    )

    highlight = st.selectbox("Highlight a neighbourhood", ["None"] + LOCS, key="mkt_highlight")
    bars = (
        alt.Chart(by_loc)
        .mark_bar(cornerRadiusEnd=7)
        .encode(
            y=alt.Y("location:N", sort="-x", title=None),
            x=alt.X("price_scaled:Q", title=f"Typical {price_title.lower()}"),
            color=alt.condition(alt.datum.location == highlight, alt.value(MARIGOLD), alt.value(SEA)),
            tooltip=[
                alt.Tooltip("location:N", title="Neighbourhood"),
                alt.Tooltip("city:N", title="City"),
                alt.Tooltip("Typical:N", title="Typical"),
            ],
        )
        .properties(height=max(420, len(by_loc) * 30))
    )
    show_chart(bars)

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




# ============================================================
# MODEL
# ============================================================
def model_view() -> None:
    st.markdown('<div class="section-title">How the estimate is made</div>', unsafe_allow_html=True)
    html("""<div class="note">There is one model for sales and one for rents. Each learns from past listings
    how size, location, age, floor and amenities move a price. Neighbourhood averages are calculated without
    peeking at the homes used for testing, so the scores below are an honest check rather than a flattering one.</div>""")
    st.write("")

    for lt in ("Sale", "Rent"):
        b = models[lt]
        st.markdown(f"### {lt} model")
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




# ============================================================
# LOAD DATA & MODELS (after the page shell is on screen, so users see
# the header and hero straight away instead of a blank spinner)
# ============================================================
def _log(msg: str) -> None:
    print(f"[propcompare] {msg}", flush=True)  # shows up in the Streamlit Cloud logs


try:
    _log("loading data…")
    with st.spinner("Preparing the demo data and pricing models. The first visit can take a minute."):
        df = load_data()
        _log("data ready, training models…")
        models = load_models()
    _log("models ready")
except Exception as exc:  # noqa: BLE001 — show a readable message instead of a blank page
    _log(f"FAILED: {exc!r}")
    st.error(
        "We couldn't load the pricing data or models. Check that the `propcompare` "
        "package is installed, then refresh the page."
    )
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()


# ============================================================
# SHOW ONLY THE ACTIVE SECTION (the other three cost nothing)
# ============================================================
if section == SECTIONS[0]:
    estimate_view(listing)
elif section == SECTIONS[1]:
    compare_view(listing)
elif section == SECTIONS[2]:
    market_view(listing)
else:
    model_view()


# ============================================================
# FOOTER
# ============================================================
html("""
<div class="footer">
    PropCompare · Mumbai &amp; Navi Mumbai<br>
    Built for exploring, not for financial or real estate advice.<br>
    <a href="#top">Back to top</a>
</div>
""")
