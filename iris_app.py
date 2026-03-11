import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

model   = joblib.load("iris_model.pkl")
encoder = joblib.load("lable_encoder.pkl")

FLOWER_IMAGES = {
    "Iris-setosa": {
        "url": "https://i.postimg.cc/Lsny6sd2/iris-setosa.png",
        "desc": "Small & delicate with broad petals. Found in Arctic regions.",
        "color": "#a78bfa",
        "emoji": "💜",
        "info": "**Iris Setosa** is the most easily identifiable of the three species. It thrives in cold, Arctic and subarctic environments across Alaska, Canada, and Siberia.\n\n- 🌡️ **Climate:** Cold, wetland & marsh habitats\n- 📏 **Size:** Smallest of the three — short, compact petals\n- 🎨 **Colour:** Deep violet-blue with white veining\n- 🔬 **Distinguishing trait:** Very broad sepals relative to petals; almost circular\n- 🌿 **Fun fact:** Setosa is *linearly separable* from the other two species — making it the easiest for ML models to classify!"
    },
    "Iris-versicolor": {
        "url": "https://i.postimg.cc/Dz8CyzVy/iris-versicolor.png",
        "desc": "Medium-sized with blue-violet petals. Native to North America.",
        "color": "#60a5fa",
        "emoji": "💙",
        "info": "**Iris Versicolor**, also called the *Blue Flag Iris*, is native to eastern North America and commonly found along riverbanks, meadows, and wet prairies.\n\n- 🌡️ **Climate:** Temperate, moist meadows & wetlands\n- 📏 **Size:** Medium — intermediate between Setosa and Virginica\n- 🎨 **Colour:** Blue-violet with yellow and white markings at the base\n- 🔬 **Distinguishing trait:** Petals narrower than Setosa but shorter than Virginica\n- 🌿 **Fun fact:** Versicolor is the provincial flower of Quebec, Canada 🍁"
    },
    "Iris-virginica": {
        "url": "https://i.postimg.cc/d0LWt0MQ/iris-virginica.png",
        "desc": "Large & elegant with deep purple flowers. Found in eastern USA.",
        "color": "#f472b6",
        "emoji": "🩷",
        "info": "**Iris Virginica**, or the *Virginia Blue Flag*, is the largest and most striking of the trio. It grows in freshwater marshes and along the coasts of the southeastern United States.\n\n- 🌡️ **Climate:** Warm, humid coastal & freshwater marshes\n- 📏 **Size:** Largest — longest petals and sepals of the three\n- 🎨 **Colour:** Rich purple-pink with intricate veining\n- 🔬 **Distinguishing trait:** Longest petal length (often >5 cm); deep, vivid colour\n- 🌿 **Fun fact:** Virginica is the hardest species for ML models to distinguish from Versicolor due to overlapping measurements!"
    },
}

BG_URL = "https://i.postimg.cc/MH8zx4Fy/bg-iris.jpg"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');

.stApp {{
    background-image:
        linear-gradient(135deg, rgba(15,10,26,0.88) 0%, rgba(30,16,48,0.82) 100%),
        url("{BG_URL}");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    font-family: 'DM Sans', sans-serif;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

/* ── FORCE SIDEBAR OPEN & NO SCROLL ── */
[data-testid="stSidebar"] {{
    transform: none !important;
    display: block !important;
    visibility: visible !important;
    width: 21rem !important;
    min-width: 21rem !important;
    background: rgba(15,10,26,0.85) !important;
    border-right: 1px solid rgba(167,139,250,0.2) !important;
    backdrop-filter: blur(20px) !important;
    overflow: hidden !important;
}}
[data-testid="stSidebar"] > div {{
    overflow: hidden !important;
    height: 100% !important;
}}
[data-testid="stSidebarContent"] {{
    overflow: hidden !important;
}}
[data-testid="stSidebar"] * {{
    color: #e9d5ff !important;
}}
/* Hide the collapse/arrow button */
[data-testid="collapsedControl"],
button[kind="header"] {{
    display: none !important;
}}

.sidebar-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem;
    color: #c084fc !important;
    letter-spacing: 0.06em;
    margin-bottom: 0.2rem;
}}

.hero {{
    text-align: center;
    padding: 2.4rem 1rem 1.2rem;
    animation: fadeDown 0.8s ease both;
}}
.hero h1 {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.4rem;
    font-weight: 300;
    letter-spacing: 0.04em;
    color: #e9d5ff;
    margin: 0;
    line-height: 1.15;
}}
.hero .subtitle {{
    font-size: 0.95rem;
    color: #c4b5fd88;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}}
.hero .byline {{
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: #a78bfa;
    margin-top: 0.6rem;
}}

.glass-card {{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(167,139,250,0.18);
    border-radius: 20px;
    padding: 2rem 2rem 1.6rem;
    backdrop-filter: blur(14px);
    margin: 1rem 0;
    animation: fadeUp 0.7s ease both;
}}

.result-card {{
    border-radius: 18px;
    padding: 1.6rem 2rem;
    text-align: center;
    backdrop-filter: blur(12px);
    animation: popIn 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
}}
.result-card h2 {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.6rem;
    font-weight: 300;
    margin: 0.2rem 0;
}}
.result-card .species-label {{
    font-size: 0.78rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.3rem;
}}
.result-card .flower-desc {{
    font-size: 0.9rem;
    opacity: 0.75;
    margin-top: 0.5rem;
    font-style: italic;
}}

.conf-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0.35rem 0;
    font-size: 0.85rem;
    color: #e9d5ff;
}}
.conf-bar-bg {{
    flex: 1;
    height: 7px;
    background: rgba(255,255,255,0.1);
    border-radius: 99px;
    overflow: hidden;
}}
.conf-bar-fill {{
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}}
.conf-label {{
    width: 120px;
    font-size: 0.82rem;
    opacity: 0.8;
}}
.conf-pct {{
    width: 40px;
    text-align: right;
    font-size: 0.82rem;
    font-weight: 500;
}}

.stButton > button {{
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #a855f7, #c084fc);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4);
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(168,85,247,0.55);
}}

.info-box {{
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    backdrop-filter: blur(12px);
    margin-top: 1rem;
    animation: fadeUp 0.6s ease both;
    line-height: 1.8;
    color: #e9d5ff;
}}

@keyframes fadeDown {{
    from {{ opacity: 0; transform: translateY(-24px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes popIn {{
    from {{ opacity: 0; transform: scale(0.88); }}
    to   {{ opacity: 1; transform: scale(1); }}
}}

[data-testid="stSlider"] [role="slider"] {{
    background: #a855f7 !important;
    border-color: #c084fc !important;
}}
</style>
""", unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="subtitle">Machine Learning · Botanical Classification</div>
    <h1>Iris Flower Prediction</h1>
    <div class="byline">by Fatima Mustafa H</div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🌿 Measurements</div>', unsafe_allow_html=True)
    st.caption("Adjust the sliders to enter flower dimensions")

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width  = st.slider("Sepal Width  (cm)", 2.0, 4.5, 3.5, 0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width  = st.slider("Petal Width  (cm)", 0.1, 2.5, 0.2, 0.1)

    predict_btn = st.button("✨ Predict Species ✨")

    st.markdown("""
    <div style='margin-top:0.8rem; font-size:0.73rem; opacity:0.4; line-height:1.5;'>
    Model: Random Forest Classifier<br>
    Dataset: UCI Iris (150 samples)<br>
    Built by Fatima Mustafa H
    </div>
    """, unsafe_allow_html=True)

# ─── Before prediction: dataset info card ────────────────────────────────────
if not predict_btn:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size:0.72rem; letter-spacing:0.2em; text-transform:uppercase; opacity:0.6; margin-bottom:1rem;">🌿 About the Iris Dataset</div>
        <div style="font-family:'Cormorant Garamond',serif; font-size:1.4rem; color:#e9d5ff; margin-bottom:0.8rem;">
            One of the most famous datasets in Machine Learning
        </div>
        <div style="font-size:0.88rem; opacity:0.7; line-height:1.8;">
            Introduced by statistician <strong>Ronald Fisher</strong> in 1936, the UCI Iris dataset contains <strong>150 samples</strong> across three species —
            <span style="color:#a78bfa;">Iris Setosa</span>, <span style="color:#60a5fa;">Iris Versicolor</span>, and <span style="color:#f472b6;">Iris Virginica</span> —
            each measured by sepal length, sepal width, petal length, and petal width.
        </div>
        <div style="display:flex; gap:1rem; margin-top:1.2rem; flex-wrap:wrap;">
            <div style="flex:1; min-width:100px; text-align:center; padding:0.8rem; background:rgba(167,139,250,0.08); border-radius:12px; border:1px solid rgba(167,139,250,0.15);">
                <div style="font-size:1.6rem; font-weight:600; color:#c084fc;">150</div>
                <div style="font-size:0.72rem; opacity:0.55; letter-spacing:0.1em; text-transform:uppercase;">Samples</div>
            </div>
            <div style="flex:1; min-width:100px; text-align:center; padding:0.8rem; background:rgba(167,139,250,0.08); border-radius:12px; border:1px solid rgba(167,139,250,0.15);">
                <div style="font-size:1.6rem; font-weight:600; color:#c084fc;">3</div>
                <div style="font-size:0.72rem; opacity:0.55; letter-spacing:0.1em; text-transform:uppercase;">Species</div>
            </div>
            <div style="flex:1; min-width:100px; text-align:center; padding:0.8rem; background:rgba(167,139,250,0.08); border-radius:12px; border:1px solid rgba(167,139,250,0.15);">
                <div style="font-size:1.6rem; font-weight:600; color:#c084fc;">4</div>
                <div style="font-size:0.72rem; opacity:0.55; letter-spacing:0.1em; text-transform:uppercase;">Features</div>
            </div>
            <div style="flex:1; min-width:100px; text-align:center; padding:0.8rem; background:rgba(167,139,250,0.08); border-radius:12px; border:1px solid rgba(167,139,250,0.15);">
                <div style="font-size:1.6rem; font-weight:600; color:#c084fc;">1936</div>
                <div style="font-size:0.72rem; opacity:0.55; letter-spacing:0.1em; text-transform:uppercase;">Published</div>
            </div>
        </div>
        <div style="margin-top:1rem; font-size:0.82rem; opacity:0.45; text-align:center; letter-spacing:0.08em;">
            Adjust the measurements in the sidebar, then hit <strong>✨Predict Species✨</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── After prediction ─────────────────────────────────────────────────────────
if predict_btn:
    features     = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_df  = pd.DataFrame(features,
                       columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

    pred_encoded  = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    species       = encoder.inverse_transform([pred_encoded])[0]
    confidence    = probabilities[pred_encoded] * 100
    info          = FLOWER_IMAGES[species]

    # ── Floating petals: inject canvas into parent document via iframe escape ────
    import streamlit.components.v1 as components

    FLOWER_URLS = {
        "Iris-setosa":     "https://i.postimg.cc/prnFrpR8/purple-flower.png",
        "Iris-versicolor": "https://i.postimg.cc/0Q4S9pgY/blue-flower.png",
        "Iris-virginica":  "https://i.postimg.cc/8cr6cjNv/pink-flower.png",
    }
    _flower_url = FLOWER_URLS[species]

    components.html(f"""
    <!DOCTYPE html><html><body style="margin:0;background:transparent;">
    <script>
    (function() {{
        const parentDoc = window.parent.document;

        const old = parentDoc.getElementById('st-petal-canvas');
        if (old) old.remove();

        const canvas = parentDoc.createElement('canvas');
        canvas.id = 'st-petal-canvas';
        Object.assign(canvas.style, {{
            position:      'fixed',
            top:           '0',
            left:          '0',
            width:         '100vw',
            height:        '100vh',
            pointerEvents: 'none',
            zIndex:        '99999',
        }});
        parentDoc.body.appendChild(canvas);

        const W = window.parent.innerWidth;
        const H = window.parent.innerHeight;
        canvas.width  = W;
        canvas.height = H;
        const ctx = canvas.getContext('2d');

        const COUNT = 50;
        const petals = [];

        // Pre-load the flower image
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.src = '{_flower_url}';

        img.onload = function() {{
            for (let i = 0; i < COUNT; i++) {{
                petals.push({{
                    x:         Math.random() * W,
                    y:         -20 - Math.random() * 400,
                    size:      28 + Math.random() * 28,
                    speedY:    1.8 + Math.random() * 2.5,
                    speedX:    (Math.random() - 0.5) * 1.5,
                    angle:     Math.random() * Math.PI * 2,
                    spin:      (Math.random() - 0.5) * 0.05,
                    opacity:   0.75 + Math.random() * 0.25,
                    wobble:    Math.random() * Math.PI * 2,
                    wobbleSpd: 0.025 + Math.random() * 0.025,
                }});
            }}

            let frame = 0;
            const MAX = 300;

            function draw() {{
                ctx.clearRect(0, 0, W, H);
                const fade = Math.max(0, 1 - frame / MAX);
                petals.forEach(p => {{
                    ctx.save();
                    ctx.globalAlpha = p.opacity * fade;
                    ctx.translate(p.x, p.y);
                    ctx.rotate(p.angle);
                    ctx.drawImage(img, -p.size / 2, -p.size / 2, p.size, p.size);
                    ctx.restore();
                    p.y      += p.speedY;
                    p.x      += p.speedX + Math.sin(p.wobble) * 0.8;
                    p.angle  += p.spin;
                    p.wobble += p.wobbleSpd;
                }});
                frame++;
                if (frame < MAX) {{
                    requestAnimationFrame(draw);
                }} else {{
                    canvas.remove();
                }}
            }}
            draw();
        }};

        img.onerror = function() {{
            // Fallback: if image fails to load, silently skip the animation
            canvas.remove();
        }};
    }})();
    </script>
    </body></html>
    """, height=0, scrolling=False)

    # Result banner
    st.markdown(f"""
    <div class="result-card" style="background: linear-gradient(135deg, {info['color']}22, {info['color']}11);
         border: 1px solid {info['color']}55;">
        <div class="species-label">Predicted Species</div>
        <h2 style="color:{info['color']};">{info['emoji']} {species.split('-')[1]}</h2>
        <div style="font-size:1.05rem; color:#e9d5ff; opacity:0.8;">{species}</div>
        <div class="flower-desc">{info['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Image + confidence
    col_img, col_stats = st.columns([1, 1])

    with col_img:
        st.image(info["url"], use_container_width=True, caption=f"{info['emoji']} {species}")

    with col_stats:
        st.markdown('<div class="glass-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:1rem;">
            <div style="font-size:0.75rem; letter-spacing:0.18em; text-transform:uppercase; opacity:0.6;">Confidence Score</div>
            <div style="font-size:2.8rem; font-weight:600; color:{info['color']}; line-height:1.1;">{confidence:.1f}%</div>
            <div style="font-size:0.8rem; opacity:0.55;">{"🟢 High confidence" if confidence > 80 else "🟡 Moderate confidence" if confidence > 50 else "🔴 Low confidence"}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.8rem; opacity:0.6; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.6rem;'>All Probabilities</div>", unsafe_allow_html=True)

        bar_colors = {"Iris-setosa": "#a78bfa", "Iris-versicolor": "#60a5fa", "Iris-virginica": "#f472b6"}
        for cls, prob in zip(encoder.classes_, probabilities):
            pct   = prob * 100
            clr   = bar_colors.get(cls, "#c084fc")
            short = cls.split("-")[1]
            st.markdown(f"""
            <div class="conf-row">
                <div class="conf-label">{short}</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{pct:.1f}%; background:{clr};"></div>
                </div>
                <div class="conf-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Species info box
    st.markdown(f"""
    <div class="info-box" style="background: linear-gradient(135deg, {info['color']}18, {info['color']}08);
         border: 1px solid {info['color']}44;">
        <div style="font-size:0.72rem; letter-spacing:0.2em; text-transform:uppercase;
             opacity:0.6; margin-bottom:0.8rem;">🔍 About this Species</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(info["info"])

    # Input summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📋 Your Input Measurements")
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, label, val in zip(
        [mc1, mc2, mc3, mc4],
        ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
        [sepal_length, sepal_width, petal_length, petal_width]
    ):
        with col:
            st.markdown(f"""
            <div style="text-align:center; padding:0.6rem; background:rgba(167,139,250,0.08);
                        border-radius:10px; border:1px solid rgba(167,139,250,0.15);">
                <div style="font-size:0.72rem; opacity:0.55; letter-spacing:0.1em; text-transform:uppercase;">{label}</div>
                <div style="font-size:1.5rem; font-weight:500; color:#c084fc;">{val}</div>
                <div style="font-size:0.72rem; opacity:0.5;">cm</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; padding-bottom:2rem;
     font-size:0.78rem; opacity:0.35; letter-spacing:0.08em;">
    🌸 Iris Prediction App &nbsp;·&nbsp; Built by <strong>Fatima Mustafa H</strong> &nbsp;·&nbsp; Powered by Streamlit
</div>
""", unsafe_allow_html=True)