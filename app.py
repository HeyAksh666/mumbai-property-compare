import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM CSS (Modern UI Setup)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Mumbai Property Compare",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern card-based UI
st.markdown("""
<style>
    /* Global Styles */
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Header */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A202C;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #718096;
        margin-bottom: 2rem;
    }

    /* Property Card Design */
    .prop-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
        transition: transform 0.2s ease-in-out;
    }
    .prop-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .prop-img-container {
        position: relative;
        width: 100%;
        height: 200px;
        overflow: hidden;
    }
    .prop-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .price-badge {
        position: absolute;
        top: 12px;
        right: 12px;
        background: rgba(15, 23, 42, 0.85);
        color: #FFFFFF;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        backdrop-filter: blur(4px);
    }

    .card-content {
        padding: 16px;
    }
    .prop-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .prop-location {
        font-size: 0.85rem;
        color: #64748B;
        margin-bottom: 12px;
    }

    .specs-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        background: #F8FAFC;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 12px;
        text-align: center;
    }
    .spec-item {
        font-size: 0.78rem;
        color: #475569;
    }
    .spec-val {
        font-weight: 600;
        color: #0F172A;
        display: block;
    }

    .amenities-tag {
        display: inline-block;
        background: #E2E8F0;
        color: #334155;
        font-size: 0.72rem;
        padding: 3px 8px;
        border-radius: 4px;
        margin-right: 4px;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. SAMPLE REALISTIC DATASET (Mumbai Real Estate)
# ---------------------------------------------------------
PROPERTIES = [
    {
        "id": 1,
        "name": "Lodha World View",
        "location": "Lower Parel, Mumbai South",
        "price_cr": 8.5,
        "price_sqft": 42500,
        "bhk": "3 BHK",
        "area_sqft": 2000,
        "possession": "Ready to Move",
        "builder": "Lodha Group",
        "image": "https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?auto=format&fit=crop&w=800&q=80",
        "amenities": ["Sea View", "Private Elevator", "Clubhouse", "Infinity Pool"]
    },
    {
        "id": 2,
        "name": "Rustomjee Elements",
        "location": "Juhu, Mumbai Western Suburbs",
        "price_cr": 12.0,
        "price_sqft": 48000,
        "bhk": "4 BHK",
        "area_sqft": 2500,
        "possession": "Under Construction (2027)",
        "builder": "Rustomjee",
        "image": "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?auto=format&fit=crop&w=800&q=80",
        "amenities": ["Private Deck", "Spa", "Concierge", "Gym"]
    },
    {
        "id": 3,
        "name": "Oberoi Sky City",
        "location": "Borivali East, Mumbai North",
        "price_cr": 3.2,
        "price_sqft": 24600,
        "bhk": "3 BHK",
        "area_sqft": 1300,
        "possession": "Ready to Move",
        "builder": "Oberoi Realty",
        "image": "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?auto=format&fit=crop&w=800&q=80",
        "amenities": ["Metro Connectivity", "Park View", "Sports Complex"]
    },
    {
        "id": 4,
        "name": "Godrej Horizon",
        "location": "Wadala, Mumbai Central",
        "price_cr": 2.8,
        "price_sqft": 25400,
        "bhk": "2 BHK",
        "area_sqft": 1100,
        "possession": "Under Construction (2026)",
        "builder": "Godrej Properties",
        "image": "https://images.unsplash.com/photo-1580587771525-78b9dba3b914?auto=format&fit=crop&w=800&q=80",
        "amenities": ["Sky Lounge", "Squash Court", "Kid's Play Area"]
    }
]

# ---------------------------------------------------------
# 3. SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.title("📍 Property Filters")

selected_location = st.sidebar.multiselect(
    "Select Region",
    options=list(set([p["location"].split(", ")[1] for p in PROPERTIES])),
    default=None,
    placeholder="All Regions"
)

selected_bhk = st.sidebar.multiselect(
    "Configuration (BHK)",
    options=sorted(list(set([p["bhk"] for p in PROPERTIES]))),
    default=None,
    placeholder="All Configurations"
)

price_range = st.sidebar.slider(
    "Budget Range (₹ Crores)",
    min_value=1.0,
    max_value=20.0,
    value=(2.0, 15.0),
    step=0.5
)

# Filter logic
filtered_props = []
for p in PROPERTIES:
    region = p["location"].split(", ")[1]
    loc_match = not selected_location or region in selected_location
    bhk_match = not selected_bhk or p["bhk"] in selected_bhk
    price_match = price_range[0] <= p["price_cr"] <= price_range[1]
    
    if loc_match and bhk_match and price_match:
        filtered_props.append(p)

# ---------------------------------------------------------
# 4. MAIN LAYOUT
# ---------------------------------------------------------
st.markdown('<div class="main-title">Mumbai Real Estate Compare</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explore, analyze, and compare premium residential projects in Mumbai side-by-side.</div>', unsafe_allow_html=True)

# Tabs for visual switching
tab_listings, tab_compare = st.tabs(["🏡 Property Catalog", "📊 Side-by-Side Comparison"])

# --- TAB 1: PROPERTY CATALOG ---
with tab_listings:
    st.write(f"Showing **{len(filtered_props)}** available properties based on your criteria:")
    
    cols = st.columns(2)  # 2-column grid layout
    
    for idx, prop in enumerate(filtered_props):
        with cols[idx % 2]:
            st.markdown(f"""
            <div class="prop-card">
                <div class="prop-img-container">
                    <img src="{prop['image']}" class="prop-img" alt="{prop['name']}">
                    <div class="price-badge">₹ {prop['price_cr']} Cr</div>
                </div>
                <div class="card-content">
                    <div class="prop-title">{prop['name']}</div>
                    <div class="prop-location">📍 {prop['location']}</div>
                    
                    <div class="specs-grid">
                        <div class="spec-item">Config<span class="spec-val">{prop['bhk']}</span></div>
                        <div class="spec-item">Carpet Area<span class="spec-val">{prop['area_sqft']} sq.ft</span></div>
                        <div class="spec-item">Avg Price<span class="spec-val">₹ {prop['price_sqft']:,}/sqft</span></div>
                    </div>
                    
                    <div>
                        {"".join([f'<span class="amenities-tag">{a}</span>' for a in prop['amenities']])}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 2: SIDE-BY-SIDE COMPARISON ---
with tab_compare:
    st.subheader("Select Properties to Compare")
    
    selected_ids = st.multiselect(
        "Choose properties from filtered list:",
        options=[p["id"] for p in filtered_props],
        format_func=lambda x: next(p["name"] for p in PROPERTIES if p["id"] == x),
        default=[p["id"] for p in filtered_props[:2]] if len(filtered_props) >= 2 else []
    )
    
    compare_props = [p for p in PROPERTIES if p["id"] in selected_ids]
    
    if compare_props:
        comp_cols = st.columns(len(compare_props))
        
        for idx, prop in enumerate(compare_props):
            with comp_cols[idx]:
                st.image(prop["image"], use_column_width=True)
                st.markdown(f"### {prop['name']}")
                
                # Metric display matrix
                st.metric("Price", f"₹ {prop['price_cr']} Cr")
                st.metric("Price per Sq.Ft.", f"₹ {prop['price_sqft']:,}")
                st.metric("Carpet Area", f"{prop['area_sqft']} sq.ft")
                st.write(**Developer:**, prop['builder'])
                st.write(**Possession:**, prop['possession'])
                st.write(**Location:**, prop['location'])
                
                st.write(**Amenities:**)
                for item in prop['amenities']:
                    st.markdown(f"- {item}")
    else:
        st.info("Select at least 2 properties to generate a side-by-side comparison matrix.")
