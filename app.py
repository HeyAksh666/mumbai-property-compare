import altair as alt
import pandas as pd
import streamlit as st
 
from propcompare.data import FURNISHING, LOCATIONS, PROPERTY_TYPES, clean, generate_dataset
from propcompare.model import predict, train_all
from propcompare.utils import fmt, per_sqft
 
st.set_page_config(page_title="PropCompare — Mumbai Property Intelligence", page_icon="🏙️", layout="wide")
 
# Replace these with licensed/local assets before public deployment.
HERO_IMAGE = "https://propertycloud.in/assets/images/gallery/1744279394.webp"
INTERIOR_IMAGE = "https://assets.architecturaldigest.in/photos/60083a17a87939f78414ee78/4%3A3/w_1600,h_1200,c_limit/asa-apartment-mumbai-featured-image-1366x768.jpg"
 
@st.cache_data(show_spinner=False)
def load_data():
    return clean(generate_dataset())
 
@st.cache_resource(show_spinner="Training valuation models…")
def load_models():
    return train_all(load_data())
 
df = load_data()
models = load_models()
LOCS = sorted(LOCATIONS)
 
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root {{--ink:#111827;--muted:#64748b;--line:rgba(15,23,42,.09);--blue:#2563eb;--paper:rgba(255,255,255,.9)}}
html {{scroll-behavior:smooth}}
body,[class*="css"] {{font-family:"DM Sans",sans-serif}}
.stApp {{background:radial-gradient(circle at 0 0,rgba(37,99,235,.08),transparent 27%),radial-gradient(circle at 100% 0,rgba(124,58,237,.07),transparent 25%),linear-gradient(180deg,#f8fafc,#fff 45%,#f8fafc);color:var(--ink)}}
.block-container {{max-width:1480px;padding-top:1.4rem;padding-bottom:5rem}}
.hero {{position:relative;min-height:430px;overflow:hidden;border-radius:32px;margin-bottom:22px;background:linear-gradient(100deg,rgba(8,15,31,.92),rgba(8,15,31,.68),rgba(8,15,31,.15)),url('{HERO_IMAGE}') center/cover;box-shadow:0 24px 70px rgba(15,23,42,.16);animation:heroIn .7s ease both}}
.hero-content {{position:relative;z-index:2;padding:52px;max-width:790px;color:#fff}}
.kicker,.badge {{text-transform:uppercase;letter-spacing:.12em;font-weight:800;font-size:.7rem}}
.kicker {{color:#bfdbfe;margin-bottom:13px}}
.hero-title {{font-family:"Manrope",sans-serif;font-size:clamp(2.5rem,5.2vw,5rem);line-height:.96;letter-spacing:-.055em;font-weight:800}}
.hero-copy {{margin-top:20px;max-width:670px;color:rgba(255,255,255,.82);line-height:1.65}}
.pills {{display:flex;flex-wrap:wrap;gap:8px;margin-top:25px}}
.pill {{padding:8px 12px;border-radius:999px;background:rgba(255,255,255,.11);border:1px solid rgba(255,255,255,.18);backdrop-filter:blur(12px);font-size:.76rem}}
.bento {{display:grid;grid-template-columns:1.25fr .75fr .75fr;grid-template-rows:150px 150px;gap:12px;margin:10px 0 28px}}
.card {{position:relative;overflow:hidden;padding:22px;border-radius:22px;border:1px solid var(--line);background:var(--paper);box-shadow:0 10px 32px rgba(15,23,42,.055);transition:transform .28s ease,box-shadow .28s ease,border-color .28s ease}}
.card:hover {{transform:translateY(-5px) perspective(1000px) rotateX(.5deg);box-shadow:0 22px 50px rgba(15,23,42,.11);border-color:rgba(37,99,235,.18)}}
.bento-main {{grid-row:1/3;background:linear-gradient(180deg,rgba(8,15,31,.04),rgba(8,15,31,.84)),url('{INTERIOR_IMAGE}') center/cover;color:#fff;display:flex;align-items:flex-end}}
.eyebrow,.kpi-label {{font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;font-weight:800;color:var(--muted)}}
.bento-main .eyebrow,.bento-main .value,.bento-main .note {{color:#fff}}
.value {{font-family:"Manrope",sans-serif;font-size:1.5rem;font-weight:800;letter-spacing:-.04em;margin-top:7px}}
.note,.kpi-sub {{color:var(--muted);font-size:.76rem;margin-top:5px}}
.kpi {{padding:19px 20px;border-radius:18px;border:1px solid var(--line);background:rgba(255,255,255,.86);box-shadow:0 8px 28px rgba(15,23,42,.045);transition:transform .22s ease,box-shadow .22s ease}}
.kpi:hover {{transform:translateY(-3px);box-shadow:0 15px 36px rgba(15,23,42,.08)}}
.kpi-value {{font-family:"Manrope",sans-serif;font-size:1.5rem;font-weight:800;letter-spacing:-.035em;margin-top:6px}}
.section {{margin:34px 0 14px}}
.section-title {{font-family:"Manrope",sans-serif;font-size:1.45rem;font-weight:800;letter-spacing:-.035em}}
.section-sub {{color:var(--muted);font-size:.87rem;margin-top:3px}}
.badge {{display:inline-block;padding:5px 10px;border-radius:999px;color:#1d4ed8;background:#eff6ff}}
div[data-baseweb="select"]>div,div[data-testid="stNumberInput"]>div {{border-radius:12px!important;border-color:rgba(15,23,42,.1)!important;background:rgba(255,255,255,.8)!important}}
button[data-baseweb="tab"] {{font-weight:700!important}}button[data-baseweb="tab"][aria-selected="true"] {{color:#2563eb!important}}
/* New button styles for minimal skeuomorphism and smooth 3D effects */
.stButton>button {{background:linear-gradient(145deg,#f0f0f3,#cacaca);border-radius:12px;border:1px solid #b8b8b8;box-shadow:5px 5px 10px #a3a3a3,-5px -5px 10px #ffffff;color:#111827;font-weight:600;padding:.5rem 1.2rem;cursor:pointer;transition:all .3s ease;font-family:"DM Sans",sans-serif;outline-offset:4px}}
.stButton>button:hover {{background:linear-gradient(145deg,#e2e2e5,#d1d1d5);box-shadow:8px 8px 15px #9a9a9a,-8px -8px 15px #ffffff;color:#2563eb;transform:translateY(-3px)}}
.stButton>button:focus-visible {{outline:3px solid #2563eb;outline-offset:3px}}
.stButton>button:active {{box-shadow:inset 3px 3px 6px #a3a3a3,inset -3px -3px 6px #ffffff;transform:translateY(1px)}}
@keyframes heroIn {{from{{opacity:0;transform:translateY(12px) scale(.985)}}to{{opacity:1;transform:none}}}}
@media(max-width:850px){{.hero-content{{padding:32px 25px}}.bento{{grid-template-columns:1fr 1fr;grid-template-rows:210px 130px 130px}}.bento-main{{grid-column:1/3;grid-row:1/2}}}}
@media(max-width:560px){{.bento{{display:block}}.card{{margin-bottom:10px;min-height:125px}}}}
#MainMenu,footer{{visibility:hidden}}header[data-testid="stHeader"]{{background:transparent}}
</style>
""", unsafe_allow_html=True)
 
def kpi(label, value, sub=""):
    return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'
 
def profile_score(p):
    score=50+min(p["parking"]*6,12)+min(p["lift"]*5,5)+min(p["gym"]*5,5)+min(p["pool"]*4,4)+min(p["security"]*5,5)
    score+=max(0,10-p["metro_dist_km"]*4)+max(0,5-p["school_dist_km"])-min(p["age_years"]*.25,8)
    return max(0,min(100,round(score)))
 
def property_form(key, d):
    st.markdown('<div class="section"><div class="section-title">Property details</div><div class="section-sub">Attributes sent to the valuation model.</div></div>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    loc=c1.selectbox("Location",LOCS,index=LOCS.index(d["location"]),key=f"{key}_loc")
    ptype=c2.selectbox("Property type",PROPERTY_TYPES,index=PROPERTY_TYPES.index(d["property_type"]),key=f"{key}_pt")
    c1,c2,c3=st.columns(3)
    bhk=c1.number_input("BHK",1,6,d["bhk"],key=f"{key}_bhk")
    bath=c2.number_input("Bathrooms",1,8,d["bathrooms"],key=f"{key}_bath")
    area=c3.number_input("Area (sq ft)",250,6000,d["area_sqft"],step=25,key=f"{key}_area")
    c1,c2,c3=st.columns(3)
    total=c1.number_input("Total floors",1,60,d["total_floors"],key=f"{key}_tf")
    floor=c2.number_input("Floor",0,int(total),min(d["floor"],int(total)),key=f"{key}_fl")
    age=c3.number_input("Property age",0.0,50.0,float(d["age_years"]),step=.5,key=f"{key}_age")
    c1,c2,c3=st.columns(3)
    furn=c1.selectbox("Furnishing",FURNISHING,index=d["furn_idx"],key=f"{key}_fu")
    parking=c2.selectbox("Parking spots",[0,1,2],index=d["parking"],key=f"{key}_pk")
    metro=c3.number_input("Metro distance (km)",0.0,15.0,float(d["metro_dist_km"]),step=.1,key=f"{key}_mt")
    school=st.slider("School distance (km)",.3,5.0,float(d["school_dist_km"]),.1,key=f"{key}_sc")
    st.caption("Amenities")
    a1,a2,a3,a4=st.columns(4)
    lift=int(a1.checkbox("Lift",d["lift"],key=f"{key}_lift")); gym=int(a2.checkbox("Gym",d["gym"],key=f"{key}_gym")); pool=int(a3.checkbox("Pool",d["pool"],key=f"{key}_pool")); sec=int(a4.checkbox("24×7 security",d["security"],key=f"{key}_sec"))
    return dict(location=loc,city=LOCATIONS[loc][0],property_type=ptype,bhk=int(bhk),bathrooms=int(bath),area_sqft=float(area),floor=int(floor),total_floors=int(total),age_years=float(age),furnishing=furn,parking=int(parking),lift=lift,gym=gym,pool=pool,security=sec,metro_dist_km=float(metro),school_dist_km=float(school))
 
A_DEF=dict(location="Andheri West",property_type="Apartment",bhk=2,bathrooms=2,area_sqft=850,floor=8,total_floors=20,age_years=4.,furn_idx=1,parking=1,metro_dist_km=.8,school_dist_km=1.,lift=True,gym=True,pool=False,security=True)
B_DEF=dict(A_DEF,location="Kharghar",area_sqft=950,floor=10,total_floors=22,age_years=2.,metro_dist_km=1.5,pool=True)
 
st.markdown('''<div class="hero"><div class="hero-content"><div class="kicker">Mumbai Metropolitan Region · Property intelligence</div><div class="hero-title">Know what a property<br>is worth before you decide.</div><div class="hero-copy">Compare homes across Mumbai and Navi Mumbai using an explainable machine-learning valuation workflow. Explore the estimate, uncertainty, market context and the features driving the result.</div><div class="pills"><span class="pill">AI valuation</span><span class="pill">Property comparison</span><span class="pill">₹ / sq ft analysis</span><span class="pill">Market explorer</span></div></div></div>''',unsafe_allow_html=True)
 
listing=st.radio("Market mode",["Sale","Rent"],horizontal=True,label_visibility="collapsed")
tab_pred,tab_cmp,tab_mkt,tab_model=st.tabs(["💰 Estimate","⚖️ Compare","📊 Market","🧠 Model"])
 
with tab_pred:
    st.markdown('<div class="section"><div class="section-title">Start with one property</div><div class="section-sub">Build a property profile, then inspect the estimate and market context.</div></div>',unsafe_allow_html=True)
    left,right=st.columns([1.25,.95],gap="large")
    with left: prop=property_form("single",A_DEF)
    result=predict(models,listing,prop); pps=per_sqft(listing,result["estimate"],prop["area_sqft"])
    loc_sub=df[(df.location==prop["location"])&(df.listing_type==listing)]
    median=loc_sub["price"].median() if len(loc_sub) else None
    score=profile_score(prop)
    with right:
        st.markdown(f'''<div class="bento"><div class="card bento-main"><div><div class="eyebrow">Estimated {"sale value" if listing=="Sale" else "monthly rent"}</div><div class="value">{fmt(listing,result["estimate"])}</div><div class="note">≈ ₹{pps:,.0f} per sq ft · {prop["location"]}</div></div></div><div class="card"><div class="eyebrow">80% range</div><div class="value" style="font-size:1.1rem">{fmt(listing,result["low"])} – {fmt(listing,result["high"])}</div><div class="note">Model uncertainty band</div></div><div class="card"><div class="eyebrow">Local median</div><div class="value">{fmt(listing,median) if median else "N/A"}</div><div class="note">{prop["location"]}</div></div><div class="card"><div class="eyebrow">Profile score</div><div class="value">{score}/100</div><div class="note">Amenities + accessibility</div></div><div class="card"><div class="eyebrow">Property basics</div><div class="value" style="font-size:1.1rem">{prop["bhk"]} BHK · {prop["area_sqft"]:,.0f} sq ft</div><div class="note">Floor {prop["floor"]}/{prop["total_floors"]} · {prop["furnishing"]}</div></div></div>''',unsafe_allow_html=True)
        st.info("The valuation is an ML estimate, not an official property valuation. The current application uses simulated data.")
        c1,c2,c3,c4=st.columns(4)
        c1.markdown(kpi("Metro",f'{prop["metro_dist_km"]:.1f} km'),unsafe_allow_html=True); c2.markdown(kpi("Parking",str(prop["parking"]),"Spots"),unsafe_allow_html=True); c3.markdown(kpi("Security","Yes" if prop["security"] else "No","24×7 input"),unsafe_allow_html=True); c4.markdown(kpi("Age",f'{prop["age_years"]:.1f} yrs'),unsafe_allow_html=True)
 
with tab_cmp:
    st.markdown('<div class="section"><div class="section-title">Which property wins?</div><div class="section-sub">Compare predicted value, efficiency and property profile — not only the headline price.</div></div>',unsafe_allow_html=True)
    ca,cb=st.columns(2,gap="large")
    with ca: st.markdown('<span class="badge">Property A</span>',unsafe_allow_html=True); pa=property_form("A",A_DEF)
    with cb: st.markdown('<span class="badge">Property B</span>',unsafe_allow_html=True); pb=property_form("B",B_DEF)
    ra=predict(models,listing,pa); rb=predict(models,listing,pb); pps_a=per_sqft(listing,ra["estimate"],pa["area_sqft"]); pps_b=per_sqft(listing,rb["estimate"],pb["area_sqft"]); profile_a=profile_score(pa); profile_b=profile_score(pb)
    diff_pct=(rb["estimate"]/ra["estimate"]-1)*100; value_leader="A" if pps_a<pps_b else "B"; profile_leader="A" if profile_a>=profile_b else "B"
    c1,c2,c3,c4=st.columns(4)
    c1.markdown(kpi("A estimate",fmt(listing,ra["estimate"]),f"₹{pps_a:,.0f}/sq ft"),unsafe_allow_html=True); c2.markdown(kpi("B estimate",fmt(listing,rb["estimate"]),f"₹{pps_b:,.0f}/sq ft"),unsafe_allow_html=True); c3.markdown(kpi("B vs A",f"{diff_pct:+.1f}%","Estimated price difference"),unsafe_allow_html=True); c4.markdown(kpi("Value leader",f"Property {value_leader}",f"Profile leader: {profile_leader}"),unsafe_allow_html=True)
    table=pd.DataFrame({"-":["Location","Type","BHK / bath","Area","Floor","Age","Furnishing","Metro","Profile score","Estimate","80% range","₹ / sq ft"],"Property A":[pa["location"],pa["property_type"],f'{pa["bhk"]} / {pa["bathrooms"]}',f'{pa["area_sqft"]:,.0f} sq ft',f'{pa["floor"]}/{pa["total_floors"]}',f'{pa["age_years"]:.1f} yrs',pa["furnishing"],f'{pa["metro_dist_km"]:.1f} km',f'{profile_a}/100',fmt(listing,ra["estimate"]),f'{fmt(listing,ra["low"])} – {fmt(listing,ra["high"])}',f'₹{pps_a:,.0f}'],"Property B":[pb["location"],pb["property_type"],f'{pb["bhk"]} / {pb["bathrooms"]}',f'{pb["area_sqft"]:,.0f} sq ft',f'{pb["floor"]}/{pb["total_floors"]}',f'{pb["age_years"]:.1f} yrs',pb["furnishing"],f'{pb["metro_dist_km"]:.1f} km',f'{profile_b}/100',fmt(listing,rb["estimate"]),f'{fmt(listing,rb["low"])} – {fmt(listing,rb["high"])}',f'₹{pps_b:,.0f}']})
    st.dataframe(table.astype(str),hide_index=True,width="stretch")
    chart_df=pd.DataFrame({"Property":["A","B"],"₹ per sq ft":[pps_a,pps_b]})
    st.altair_chart(alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=8,cornerRadiusTopRight=8).encode(x=alt.X("Property:N",axis=alt.Axis(labelAngle=0)),y=alt.Y("₹ per sq ft:Q",title="Estimated ₹ / sq ft"),tooltip=["Property","₹ per sq ft"]).properties(height=300,title="Price efficiency"),width="stretch")
    st.success(f"Property {value_leader} leads on estimated ₹/sq ft; Property {profile_leader} has the stronger profile score.") if value_leader==profile_leader else st.info(f"Property {value_leader} is cheaper per sq ft, while Property {profile_leader} has the stronger profile score.")
 
with tab_mkt:
    st.markdown('<div class="section"><div class="section-title">Understand the market around the property</div><div class="section-sub">Explore simulated micro-market distributions instead of looking at one prediction in isolation.</div></div>',unsafe_allow_html=True)
    sub=df[df.listing_type==listing].copy(); by_loc=sub.groupby(["location","city"],as_index=False)["price"].median().sort_values("price",ascending=False); by_loc["label"]=[fmt(listing,v) for v in by_loc["price"]]
    st.altair_chart(alt.Chart(by_loc).mark_bar(cornerRadiusEnd=6).encode(y=alt.Y("location:N",sort="-x",title=None),x=alt.X("price:Q",title="Median price"),color=alt.Color("city:N",title="Market"),tooltip=["location","city","label"]).properties(height=560),width="stretch")
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="section"><div class="section-title">Area vs price</div><div class="section-sub">Interactive distribution by BHK.</div></div>',unsafe_allow_html=True)
        st.altair_chart(alt.Chart(sub).mark_circle(opacity=.55,size=55).encode(x=alt.X("area_sqft:Q",title="Area (sq ft)"),y=alt.Y("price:Q",title="Price"),color=alt.Color("bhk:O",title="BHK"),tooltip=["location","bhk","area_sqft","price"]).interactive().properties(height=410),width="stretch")
    with c2:
        st.markdown('<div class="section"><div class="section-title">Price distribution</div><div class="section-sub">See how the sample spreads by BHK.</div></div>',unsafe_allow_html=True)
        st.altair_chart(alt.Chart(sub).mark_boxplot(size=55).encode(x=alt.X("bhk:O",title="BHK"),y=alt.Y("price:Q",title="Price")).properties(height=410),width="stretch")
 
with tab_model:
    st.markdown('<div class="section"><div class="section-title">Inside the valuation engine</div><div class="section-sub">A polished product should also explain its model and limitations.</div></div>',unsafe_allow_html=True)
    st.warning("The current project trains on a generated/simulated dataset. These metrics measure model behaviour on that dataset; they are not evidence of real-world Mumbai valuation accuracy.")
    for lt in ("Sale","Rent"):
        b=models[lt]; st.markdown(f'<span class="badge">{lt} model</span>',unsafe_allow_html=True); c1,c2,c3,c4=st.columns(4)
        c1.markdown(kpi("R²",f'{b.metrics["R2"]:.3f}',"Test set"),unsafe_allow_html=True); c2.markdown(kpi("MAPE",f'{b.metrics["MAPE_%"]:.1f}%'),unsafe_allow_html=True); c3.markdown(kpi("MAE",fmt(lt,b.metrics["MAE"])),unsafe_allow_html=True); c4.markdown(kpi("Rows",f'{b.metrics["train_rows"]:,} / {b.metrics["test_rows"]:,}',"Train / test"),unsafe_allow_html=True)
        imp=b.importances.head(10).reset_index(); imp.columns=["feature","importance"]
        st.altair_chart(alt.Chart(imp).mark_bar(cornerRadiusEnd=5).encode(y=alt.Y("feature:N",sort="-x",title=None),x=alt.X("importance:Q",title="Importance"),tooltip=["feature","importance"]).properties(height=300,title=f"Top features — {lt}"),width="stretch")
    with st.expander("Preview generated dataset"): st.dataframe(df.head(100),width="stretch")
 
st.markdown('<hr style="border:0;border-top:1px solid rgba(15,23,42,.08);margin-top:55px"><div style="text-align:center;color:#64748b;font-size:.78rem;line-height:1.7;padding:18px"> <strong>PropCompare</strong> · Mumbai & Navi Mumbai Property Intelligence<br>Current valuation engine uses simulated project data. Estimates are indicative, not official valuations.<br>Replace remote imagery with licensed/local assets before public deployment.</div>',unsafe_allow_html=True)
 
