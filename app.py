"""
BANK DSS — Analytical Command Center
Decision Support System for Bank Telemarketing Campaign
Author: DSS Project | Framework: Streamlit
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from auto_setup import run_setup_if_needed
run_setup_if_needed()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit command
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank DSS | Command Center",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Analytical Command Center Theme (Purple-Teal)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&family=Orbitron:wght@400;600;700;900&display=swap');

/* ── Root Variables ── */
:root {
    --bg-base:       #0B0D17;
    --bg-card:       #111320;
    --bg-card2:      #161928;
    --bg-sidebar:    #0D0F1E;
    --purple-dark:   #1A0F3C;
    --purple:        #6C3BDB;
    --purple-light:  #9B72F5;
    --purple-glow:   rgba(108,59,219,0.3);
    --teal:          #00C8B4;
    --teal-light:    #4DFFF0;
    --teal-glow:     rgba(0,200,180,0.25);
    --teal-dark:     #00897B;
    --amber:         #F5A623;
    --red:           #FF4D6D;
    --green:         #2ECC71;
    --text-primary:  #E8EAF6;
    --text-secondary:#8892B0;
    --text-muted:    #546E7A;
    --border:        rgba(108,59,219,0.2);
    --border-teal:   rgba(0,200,180,0.2);
    --font-display:  'Orbitron', monospace;
    --font-body:     'DM Sans', sans-serif;
    --font-mono:     'Space Mono', monospace;
}

/* ── Base & Background ── */
.stApp { background-color: var(--bg-base); font-family: var(--font-body); }
.main .block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── Global Typography ── */
h1, h2, h3, h4, h5, h6 { font-family: var(--font-body); color: var(--text-primary); }
p, li, span, label { color: var(--text-secondary); }

/* ── Sidebar Logo Block ── */
.sidebar-logo {
    background: linear-gradient(135deg, var(--purple-dark) 0%, #0D1A2E 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.2rem 1rem 1rem 1rem;
    margin: 0 -1rem 1.2rem -1rem;
    text-align: center;
}
.sidebar-logo .logo-title {
    font-family: var(--font-display);
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--teal);
    letter-spacing: 0.12em;
    line-height: 1.3;
}
.sidebar-logo .logo-sub {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}
.sidebar-badge {
    display: inline-block;
    background: var(--purple-glow);
    border: 1px solid var(--purple);
    color: var(--purple-light);
    font-family: var(--font-mono);
    font-size: 0.55rem;
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
    margin-top: 0.4rem;
    letter-spacing: 0.1em;
}

/* ── Sidebar Nav Labels ── */
.nav-section {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--text-muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.6rem 0 0.3rem 0;
    border-top: 1px solid var(--border);
    margin-top: 0.5rem;
}

/* ── Radio buttons (nav) ── */
div[data-testid="stRadio"] label {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
    cursor: pointer;
    transition: color 0.2s;
}
div[data-testid="stRadio"] label:hover { color: var(--teal) !important; }
div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {
    font-size: 0.88rem;
}

/* ── Page Header ── */
.page-header {
    display: flex;
    align-items: flex-end;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.page-header .ph-icon {
    font-size: 2rem;
    line-height: 1;
}
.page-header .ph-text .ph-title {
    font-family: var(--font-display);
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.06em;
    margin: 0;
    line-height: 1.2;
}
.page-header .ph-text .ph-sub {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    margin-top: 0.15rem;
}
.ph-badge {
    margin-left: auto;
    background: var(--teal-glow);
    border: 1px solid var(--teal);
    color: var(--teal);
    font-family: var(--font-mono);
    font-size: 0.6rem;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    letter-spacing: 0.1em;
    white-space: nowrap;
    align-self: center;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.1rem 1.2rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    border-color: var(--purple-light);
    box-shadow: 0 0 16px var(--purple-glow);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--purple), var(--teal));
}
.kpi-card .kpi-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.kpi-card .kpi-value {
    font-family: var(--font-display);
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}
.kpi-card .kpi-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}
.kpi-card .kpi-delta {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 600;
}
.kpi-card .kpi-delta.up { color: var(--green); }
.kpi-card .kpi-delta.down { color: var(--red); }
.kpi-card .kpi-icon {
    position: absolute;
    top: 1rem; right: 1rem;
    font-size: 1.4rem;
    opacity: 0.3;
}
.kpi-teal::before { background: var(--teal) !important; }
.kpi-amber::before { background: var(--amber) !important; }
.kpi-red::before { background: var(--red) !important; }
.kpi-green::before { background: var(--green) !important; }

/* ── Section Cards ── */
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.2rem;
}
.section-card .sc-title {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--teal);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-teal);
}

/* ── Insight Box ── */
.insight-box {
    background: linear-gradient(135deg, rgba(108,59,219,0.1) 0%, rgba(0,200,180,0.06) 100%);
    border: 1px solid var(--border);
    border-left: 3px solid var(--teal);
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    margin: 0.6rem 0;
}
.insight-box .ib-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--teal);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.insight-box .ib-text {
    color: var(--text-primary);
    font-size: 0.88rem;
    line-height: 1.5;
}

/* ── Warning / Alert Box ── */
.alert-box {
    background: rgba(245,166,35,0.08);
    border: 1px solid rgba(245,166,35,0.3);
    border-left: 3px solid var(--amber);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
}
.alert-box .ab-label {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    color: var(--amber);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.alert-box .ab-text { color: var(--text-primary); font-size: 0.85rem; line-height: 1.4; }

/* ── Success Box ── */
.success-box {
    background: rgba(46,204,113,0.08);
    border: 1px solid rgba(46,204,113,0.3);
    border-left: 3px solid var(--green);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
}
.success-box .sb-label {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    color: var(--green);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.success-box .sb-text { color: var(--text-primary); font-size: 0.85rem; line-height: 1.4; }

/* ── Cluster Badges ── */
.cluster-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin: 0.1rem;
}
.cb-0 { background: rgba(108,59,219,0.2); border: 1px solid var(--purple); color: var(--purple-light); }
.cb-1 { background: rgba(0,200,180,0.15); border: 1px solid var(--teal); color: var(--teal-light); }
.cb-2 { background: rgba(245,166,35,0.15); border: 1px solid var(--amber); color: var(--amber); }
.cb-3 { background: rgba(46,204,113,0.15); border: 1px solid var(--green); color: var(--green); }

/* ── Prediction Result ── */
.pred-result-high {
    background: linear-gradient(135deg, rgba(46,204,113,0.12), rgba(0,200,180,0.08));
    border: 1px solid var(--green);
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
}
.pred-result-low {
    background: linear-gradient(135deg, rgba(255,77,109,0.10), rgba(108,59,219,0.08));
    border: 1px solid var(--red);
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
}
.pred-score {
    font-family: var(--font-display);
    font-size: 3rem;
    font-weight: 900;
    line-height: 1;
    margin: 0.3rem 0;
}
.pred-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Data Table ── */
.dataframe { font-size: 0.8rem !important; }
div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 6px; }

/* ── Sliders & Inputs ── */
div[data-testid="stSlider"] label { color: var(--text-secondary) !important; font-size: 0.85rem !important; }
div[data-testid="stSelectbox"] label { color: var(--text-secondary) !important; font-size: 0.85rem !important; }
div[data-testid="stNumberInput"] label { color: var(--text-secondary) !important; font-size: 0.85rem !important; }

/* ── Streamlit default overrides ── */
.stButton button {
    background: linear-gradient(135deg, var(--purple) 0%, var(--teal-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.85 !important; }
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
div[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.75rem !important; }
div[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-family: var(--font-display) !important; font-size: 1.5rem !important; }
div[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Tabs ── */
div[data-testid="stTabs"] button {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    color: var(--text-muted) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--teal) !important;
    border-bottom-color: var(--teal) !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
div[data-testid="stExpander"] summary { color: var(--text-secondary) !important; font-size: 0.85rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--purple); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--purple-light); }

/* ── Plotly chart container ── */
.js-plotly-plot .plotly { border-radius: 8px; }

/* ── Footer ── */
.footer-bar {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    text-align: center;
}
.footer-bar p {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
}

/* ── Segment Divider ── */
.seg-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans, sans-serif', color='#8892B0', size=11),
    title_font=dict(family='DM Sans, sans-serif', color='#E8EAF6', size=13),
    xaxis=dict(gridcolor='rgba(108,59,219,0.1)', linecolor='rgba(108,59,219,0.2)',
               tickfont=dict(color='#8892B0', size=10), zerolinecolor='rgba(108,59,219,0.15)'),
    yaxis=dict(gridcolor='rgba(108,59,219,0.1)', linecolor='rgba(108,59,219,0.2)',
               tickfont=dict(color='#8892B0', size=10), zerolinecolor='rgba(108,59,219,0.15)'),
    legend=dict(bgcolor='rgba(17,19,32,0.8)', bordercolor='rgba(108,59,219,0.3)',
                borderwidth=1, font=dict(color='#8892B0', size=10)),
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=['#6C3BDB', '#00C8B4', '#F5A623', '#FF4D6D', '#2ECC71', '#9B72F5', '#4DFFF0'],
)

COLORS = {
    'purple': '#6C3BDB', 'purple_light': '#9B72F5',
    'teal': '#00C8B4', 'teal_light': '#4DFFF0',
    'amber': '#F5A623', 'red': '#FF4D6D',
    'green': '#2ECC71', 'text': '#E8EAF6', 'muted': '#8892B0',
}
CLUSTER_COLORS = [COLORS['purple'], COLORS['teal'], COLORS['amber'], COLORS['green']]

# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'bank_data.csv'))
    return df

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'lr': 'models/lr_model.pkl',
        'rf': 'models/rf_model.pkl',
        'kmeans': 'models/kmeans.pkl',
        'scaler': 'models/scaler.pkl',
        'seg_scaler': 'models/seg_scaler.pkl',
        'encoders': 'models/encoders.pkl',
        'feature_cols': 'models/feature_cols.pkl',
        'seg_features': 'models/seg_features.pkl',
        'cluster_profiles': 'models/cluster_profiles.pkl',
        'feature_importance': 'models/feature_importance.pkl',
        'test_results': 'models/test_results.pkl',
    }
    base = os.path.dirname(__file__) + '/'
    for key, path in model_files.items():
        try:
            models[key] = joblib.load(base + path)
        except Exception as e:
            models[key] = None
    return models

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def make_chart(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

def render_kpi(label, value, sub='', delta=None, delta_dir='up', variant=''):
    delta_html = ''
    if delta:
        d_class = 'up' if delta_dir == 'up' else 'down'
        d_arrow = '▲' if delta_dir == 'up' else '▼'
        delta_html = f'<div class="kpi-delta {d_class}">{d_arrow} {delta}</div>'
    return f"""
    <div class="kpi-card {variant}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
        {delta_html}
    </div>"""

def page_header(icon, title, subtitle, badge=None):
    badge_html = f'<span class="ph-badge">{badge}</span>' if badge else ''
    st.markdown(f"""
    <div class="page-header">
        <div class="ph-icon">{icon}</div>
        <div class="ph-text">
            <div class="ph-title">{title}</div>
            <div class="ph-sub">{subtitle}</div>
        </div>
        {badge_html}
    </div>""", unsafe_allow_html=True)

def insight(text, label='INSIGHT'):
    st.markdown(f"""
    <div class="insight-box">
        <div class="ib-label">⬡ {label}</div>
        <div class="ib-text">{text}</div>
    </div>""", unsafe_allow_html=True)

def alert(text, label='PERHATIAN'):
    st.markdown(f"""
    <div class="alert-box">
        <div class="ab-label">⚠ {label}</div>
        <div class="ab-text">{text}</div>
    </div>""", unsafe_allow_html=True)

def success_box(text, label='REKOMENDASI'):
    st.markdown(f"""
    <div class="success-box">
        <div class="sb-label">✓ {label}</div>
        <div class="sb-text">{text}</div>
    </div>""", unsafe_allow_html=True)

def section_card_start(title):
    st.markdown(f'<div class="section-card"><div class="sc-title">{title}</div>', unsafe_allow_html=True)

def section_card_end():
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-title">BANK DSS<br>COMMAND CENTER</div>
        <div class="logo-sub">DECISION SUPPORT SYSTEM v2.0</div>
        <div class="sidebar-badge">◈ ANALYTICAL INTELLIGENCE</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="nav-section">◈ MODUL SISTEM</div>', unsafe_allow_html=True)
    page = st.radio(
        label="Navigation",
        options=[
            "🏠 Overview & Data Explorer",
            "🧩 Segmentasi Customer",
            "🎯 Prediksi Respons Campaign",
            "📊 Analisis Efektivitas Campaign",
            "💰 Simulasi & Optimasi Budget",
            "🧠 Rekomendasi Keputusan"
        ],
        label_visibility="collapsed"
    )

    st.markdown('<div class="nav-section">◈ STATUS SISTEM</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#8892B0; line-height:2;">
        <span style="color:#2ECC71;">●</span> Data Pipeline: <span style="color:#E8EAF6;">AKTIF</span><br>
        <span style="color:#2ECC71;">●</span> ML Engine: <span style="color:#E8EAF6;">LOADED</span><br>
        <span style="color:#2ECC71;">●</span> Monte Carlo: <span style="color:#E8EAF6;">READY</span><br>
        <span style="color:#F5A623;">●</span> Dataset: <span style="color:#E8EAF6;">SYNTHETIC</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="nav-section">◈ TEORI DSS</div>', unsafe_allow_html=True)
    with st.expander("📖 Kerangka Teoritis", expanded=False):
        st.markdown("""
        <div style="font-size:0.78rem; color:#8892B0; line-height:1.7;">
        <b style="color:#9B72F5;">Model Simon (1977)</b><br>
        Intelligence → Design → Choice<br><br>
        <b style="color:#00C8B4;">Komponen DSS</b><br>
        • Data Management<br>
        • Model Management<br>
        • Knowledge Engine<br>
        • User Interface<br><br>
        <b style="color:#F5A623;">Referensi</b><br>
        Moro et al. (2014)<br>
        Decision Support Systems<br>
        Elsevier, 62, 22–31
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1rem; padding-top:1rem; border-top:1px solid rgba(108,59,219,0.2);">
        <div style="font-family:'Space Mono',monospace; font-size:0.55rem; color:#546E7A; text-align:center; line-height:1.8;">
        UCI Bank Marketing Dataset<br>
        © 2026 Decision Support System
        </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = load_data()
models = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: OVERVIEW & DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview & Data Explorer":
    page_header("🏠", "OVERVIEW & DATA EXPLORER",
                "INTELLIGENCE LAYER — FASE INTELIJEN DALAM MODEL SIMON (1977)")

    # ── KPI Row ──
    cr = df['y'].mean()
    total = len(df)
    converters = df['y'].sum()
    avg_dur = df['duration'].mean()

    st.markdown(f"""
    <div class="kpi-grid">
        {render_kpi("TOTAL NASABAH", f"{total:,}", "dalam dataset", variant="kpi-teal")}
        {render_kpi("CONVERSION RATE", f"{cr:.1%}", "berlangganan deposito", delta="vs 11.3% baseline", delta_dir="up", variant="kpi-green")}
        {render_kpi("TOTAL SUBSCRIBER", f"{int(converters):,}", "nasabah terkonversi")}
        {render_kpi("RATA-RATA DURASI", f"{avg_dur:.0f}s", "durasi panggilan", variant="kpi-amber")}
        {render_kpi("FITUR TERSEDIA", "20", "variabel prediksi", variant="kpi-teal")}
    </div>""", unsafe_allow_html=True)

    insight(
        "Dataset ini mencakup kampanye telemarketing bank Portugal dengan <b>20 fitur prediktif</b> "
        "mencakup demografis nasabah, konteks kampanye, dan indikator ekonomi makro. "
        "Conversion rate <b>27.3%</b> menunjukkan ketidakseimbangan kelas yang memerlukan strategi "
        "class-balancing dalam pemodelan prediktif.",
        "KONTEKS BISNIS"
    )

    tab1, tab2, tab3 = st.tabs(["📈 DISTRIBUSI DATA", "🔗 KORELASI FITUR", "🔎 DATA EXPLORER"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Distribusi usia
            fig = px.histogram(df, x='age', nbins=30, color_discrete_sequence=[COLORS['purple']])
            fig.update_layout(**PLOTLY_LAYOUT, title='Distribusi Usia Nasabah',
                              bargap=0.05)
            fig.add_vline(x=df['age'].mean(), line_dash='dash',
                          line_color=COLORS['teal'],
                          annotation_text=f"Mean: {df['age'].mean():.0f}",
                          annotation_font_color=COLORS['teal'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Job distribution
            job_counts = df.groupby('job')['y'].agg(['count', 'mean']).reset_index()
            job_counts.columns = ['job', 'count', 'conv_rate']
            job_counts = job_counts.sort_values('count', ascending=True)
            fig = go.Figure(go.Bar(
                x=job_counts['count'], y=job_counts['job'],
                orientation='h',
                marker=dict(
                    color=job_counts['conv_rate'],
                    colorscale=[[0, COLORS['purple']], [0.5, COLORS['teal']], [1, COLORS['green']]],
                    showscale=True,
                    colorbar=dict(title='Conv Rate', tickfont=dict(color='#8892B0', size=9))
                )
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title='Distribusi Pekerjaan & Conversion Rate')
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            # Conversion by month
            month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                           'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            month_data = df.groupby('month').agg(
                conversion=('y', 'mean'), count=('y', 'count')
            ).reindex(month_order).dropna().reset_index()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=month_data['month'], y=month_data['count'],
                name='Volume', marker_color=COLORS['purple'],
                opacity=0.6
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=month_data['month'], y=month_data['conversion'] * 100,
                name='Conv Rate %', mode='lines+markers',
                line=dict(color=COLORS['teal'], width=2.5),
                marker=dict(size=7, color=COLORS['teal'])
            ), secondary_y=True)
            fig.update_layout(**PLOTLY_LAYOUT, title='Volume & Conversion per Bulan')
            fig.update_yaxes(title_text="Volume Kontak", secondary_y=False,
                             title_font=dict(color='#8892B0', size=10))
            fig.update_yaxes(title_text="Conversion Rate (%)", secondary_y=True,
                             title_font=dict(color=COLORS['teal'], size=10),
                             tickfont=dict(color=COLORS['teal']))
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            # Duration vs conversion
            df_dur = df.copy()
            df_dur['dur_bin'] = pd.cut(df_dur['duration'], bins=[0, 60, 180, 360, 600, 5000],
                                       labels=['<1min', '1-3min', '3-6min', '6-10min', '>10min'])
            dur_conv = df_dur.groupby('dur_bin', observed=True)['y'].mean().reset_index()
            fig = go.Figure(go.Bar(
                x=dur_conv['dur_bin'].astype(str), y=dur_conv['y'] * 100,
                marker=dict(
                    color=dur_conv['y'] * 100,
                    colorscale=[[0, COLORS['purple']], [1, COLORS['green']]],
                    showscale=False
                ),
                text=[f"{v:.1f}%" for v in dur_conv['y'] * 100],
                textposition='outside',
                textfont=dict(color='#E8EAF6', size=11)
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title='Conversion Rate per Durasi Panggilan')
            fig.update_yaxes(ticksuffix='%')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        num_cols = ['age', 'duration', 'campaign', 'previous', 'emp.var.rate',
                    'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']
        corr = df[num_cols].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale=[[0, COLORS['red']], [0.5, '#161928'], [1, COLORS['teal']]],
            zmid=0,
            text=np.round(corr.values, 2), texttemplate='%{text}',
            textfont=dict(size=9, color='#E8EAF6'),
            hoverongaps=False
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title='Matriks Korelasi Fitur Numerik',
                          height=450)
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        insight(
            "Korelasi paling kuat terhadap target (<b>y</b>) ditemukan pada <b>duration</b> (durasi panggilan) "
            "dan variabel ekonomi makro seperti <b>euribor3m</b> dan <b>emp.var.rate</b>. "
            "Durasi tinggi mengindikasikan nasabah lebih engaged, namun baru diketahui setelah panggilan "
            "sehingga tidak bisa digunakan sebagai prediktor pra-kampanye secara langsung.",
            "ANALISIS KORELASI"
        )

    with tab3:
        st.markdown('<div class="sc-title">🔎 FILTER & PREVIEW DATA</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            filter_job = st.multiselect("Pekerjaan", options=sorted(df['job'].unique()),
                                        default=[], placeholder="Semua pekerjaan")
        with c2:
            filter_edu = st.multiselect("Pendidikan", options=sorted(df['education'].unique()),
                                        default=[], placeholder="Semua pendidikan")
        with c3:
            filter_target = st.selectbox("Target", ["Semua", "Berlangganan (Yes)", "Tidak (No)"])

        df_filter = df.copy()
        if filter_job:
            df_filter = df_filter[df_filter['job'].isin(filter_job)]
        if filter_edu:
            df_filter = df_filter[df_filter['education'].isin(filter_edu)]
        if filter_target == "Berlangganan (Yes)":
            df_filter = df_filter[df_filter['y'] == 1]
        elif filter_target == "Tidak (No)":
            df_filter = df_filter[df_filter['y'] == 0]

        st.markdown(f"""<div style="font-family:'Space Mono',monospace; font-size:0.7rem;
                    color:#8892B0; margin-bottom:0.5rem;">
                    Menampilkan <span style="color:#00C8B4;">{len(df_filter):,}</span>
                    dari <span style="color:#E8EAF6;">{len(df):,}</span> records
                    | Conversion Rate: <span style="color:#2ECC71;">{df_filter['y'].mean():.1%}</span>
                    </div>""", unsafe_allow_html=True)
        st.dataframe(
            df_filter.head(100).style.map(
                lambda v: 'color: #2ECC71; font-weight:600' if v == 1 else
                          ('color: #FF4D6D' if v == 0 else ''),
                subset=['y']
            ),
            use_container_width=True, height=300
        )

    st.markdown('<div class="footer-bar"><p>Data Source: UCI Bank Marketing Dataset | Moro, S., Cortez, P., & Rita, P. (2014). Decision Support Systems, Elsevier, 62, 22–31.</p></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: SEGMENTASI CUSTOMER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧩 Segmentasi Customer":
    page_header("🧩", "SEGMENTASI CUSTOMER",
                "K-MEANS CLUSTERING — FASE INTELIJEN & DESAIN (SIMON, 1977)")

    cluster_names = {0: 'Skeptics', 1: 'Loyalists', 2: 'Prospects', 3: 'Champions'}
    cluster_desc = {
        0: 'Nasabah dengan engagement rendah, banyak kontak tanpa konversi. Perlu pendekatan ulang berbeda.',
        1: 'Nasabah dengan riwayat kontak positif, responsif tapi volume kecil.',
        2: 'Nasabah dengan potensi konversi tinggi berdasarkan profil ekonomi.',
        3: 'Segmen terbesar dengan conversion rate tertinggi — prioritas utama kampanye.'
    }
    cluster_strategy = {
        0: '→ Kurangi frekuensi kontak. Evaluasi ulang timing dan channel.',
        1: '→ Pertahankan relasi. Tawarkan produk premium setelah deposito.',
        2: '→ Jadikan prioritas follow-up cepat dengan skrip yang dipersonalisasi.',
        3: '→ Alokasikan budget terbesar. Manfaatkan momentum ekonomi positif.'
    }

    # Cluster overview
    cp = models['cluster_profiles']
    cols = st.columns(4)
    for i, col in enumerate(cols):
        with col:
            conv = cp.loc[i, 'conversion_rate']
            sz = cp.loc[i, 'size']
            name = cluster_names[i]
            badge_class = ['cb-0', 'cb-1', 'cb-2', 'cb-3'][i]
            conv_color = COLORS['green'] if conv > 0.28 else (COLORS['amber'] if conv > 0.22 else COLORS['red'])
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">CLUSTER {i}</div>
                <div style="font-size:0.95rem; font-weight:600; color:#E8EAF6; margin-bottom:0.3rem;">
                    <span class="cluster-badge {badge_class}">{name}</span>
                </div>
                <div style="font-family:'Orbitron',monospace; font-size:1.5rem; color:{conv_color}; font-weight:700;">
                    {conv:.1%}
                </div>
                <div class="kpi-sub">Conv Rate | n={sz:,}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="seg-divider"></div>', unsafe_allow_html=True)

    tab_a, tab_b, tab_c = st.tabs(["📊 VISUALISASI CLUSTER", "📋 PROFIL SEGMEN", "🎯 STRATEGI PER SEGMEN"])

    with tab_a:
        col1, col2 = st.columns([3, 2])
        with col1:
            # Scatter plot - Age vs Duration colored by cluster
            fig = go.Figure()
            for i in range(4):
                mask = df['cluster'] == i
                fig.add_trace(go.Scatter(
                    x=df[mask]['age'].sample(min(300, mask.sum()), random_state=42),
                    y=df[mask]['duration'].sample(min(300, mask.sum()), random_state=42),
                    mode='markers',
                    name=f"Cluster {i}: {cluster_names[i]}",
                    marker=dict(color=CLUSTER_COLORS[i], size=6, opacity=0.65,
                                line=dict(color='rgba(255,255,255,0.1)', width=0.5))
                ))
            fig.update_layout(**PLOTLY_LAYOUT,
                              title='Distribusi Cluster: Usia vs Durasi Panggilan',
                              xaxis_title='Usia (tahun)', yaxis_title='Durasi Panggilan (detik)',
                              height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Pie chart cluster distribution
            sizes = [cp.loc[i, 'size'] for i in range(4)]
            names = [f"C{i}: {cluster_names[i]}" for i in range(4)]
            fig = go.Figure(go.Pie(
                labels=names, values=sizes,
                marker_colors=CLUSTER_COLORS,
                hole=0.55,
                textfont=dict(size=10, color='#E8EAF6'),
                hoverinfo='label+percent+value'
            ))
            fig.add_annotation(text=f"<b>{sum(sizes):,}</b><br><span style='font-size:10px'>nasabah</span>",
                               x=0.5, y=0.5, font=dict(size=14, color='#E8EAF6'), showarrow=False)
            fig.update_layout(**PLOTLY_LAYOUT, title='Komposisi Segmen Customer',
                              showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Radar chart per cluster
        categories = ['Avg Age', 'Avg Duration', 'Avg Campaign', 'Conv Rate', 'Avg Previous']
        cp_norm = cp.copy()
        for col_n in ['avg_age', 'avg_duration', 'avg_campaign', 'conversion_rate', 'avg_previous']:
            col_max = cp_norm[col_n].max()
            col_min = cp_norm[col_n].min()
            rng = col_max - col_min if col_max != col_min else 1
            cp_norm[col_n] = (cp_norm[col_n] - col_min) / rng

        fig = go.Figure()
        for i in range(4):
            vals = [cp_norm.loc[i, 'avg_age'], cp_norm.loc[i, 'avg_duration'],
                    cp_norm.loc[i, 'avg_campaign'], cp_norm.loc[i, 'conversion_rate'],
                    cp_norm.loc[i, 'avg_previous']]
            vals_closed = vals + [vals[0]]
            cats_closed = categories + [categories[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats_closed,
                name=f"Cluster {i}: {cluster_names[i]}",
                line=dict(color=CLUSTER_COLORS[i], width=2),
                fill='toself',
                fillcolor=CLUSTER_COLORS[i].replace(')', ', 0.08)').replace('rgb', 'rgba') if 'rgb' in CLUSTER_COLORS[i]
                           else f"rgba({int(CLUSTER_COLORS[i][1:3],16)},{int(CLUSTER_COLORS[i][3:5],16)},{int(CLUSTER_COLORS[i][5:7],16)},0.08)"
            ))
        fig.update_layout(**PLOTLY_LAYOUT,
                          title='Profil Multidimensi per Cluster (Normalized)',
                          polar=dict(
                              radialaxis=dict(visible=True, range=[0, 1],
                                             tickfont=dict(color='#546E7A', size=8),
                                             gridcolor='rgba(108,59,219,0.15)'),
                              angularaxis=dict(tickfont=dict(color='#8892B0', size=10),
                                              gridcolor='rgba(108,59,219,0.15)'),
                              bgcolor='rgba(0,0,0,0)'
                          ),
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab_b:
        st.markdown('<div class="sc-title">PROFIL STATISTIK TIAP SEGMEN</div>', unsafe_allow_html=True)

        display_cp = pd.DataFrame({
            'Cluster': [f"C{i}: {cluster_names[i]}" for i in range(4)],
            'Jumlah': [f"{cp.loc[i,'size']:,}" for i in range(4)],
            'Conv Rate': [f"{cp.loc[i,'conversion_rate']:.1%}" for i in range(4)],
            'Rata2 Usia': [f"{cp.loc[i,'avg_age']:.1f} thn" for i in range(4)],
            'Rata2 Durasi': [f"{cp.loc[i,'avg_duration']:.0f} dtk" for i in range(4)],
            'Rata2 Kampanye': [f"{cp.loc[i,'avg_campaign']:.1f}x" for i in range(4)],
            'Rata2 Prev Contact': [f"{cp.loc[i,'avg_previous']:.1f}x" for i in range(4)],
        })
        st.dataframe(display_cp, use_container_width=True, hide_index=True)

        st.markdown('<div class="seg-divider"></div>', unsafe_allow_html=True)

        # Per cluster detail
        for i in range(4):
            badge_class = ['cb-0', 'cb-1', 'cb-2', 'cb-3'][i]
            conv = cp.loc[i, 'conversion_rate']
            conv_color = COLORS['green'] if conv > 0.28 else (COLORS['amber'] if conv > 0.22 else COLORS['red'])
            with st.expander(f"Cluster {i}: {cluster_names[i]} — {cp.loc[i, 'size']:,} nasabah | Conv: {conv:.1%}"):
                st.markdown(f"""
                <div style="color:#E8EAF6; font-size:0.88rem; line-height:1.7;">
                    <b style="color:{CLUSTER_COLORS[i]};">Deskripsi:</b> {cluster_desc[i]}<br><br>
                    <b style="color:{conv_color};">Conversion Rate:</b> {conv:.1%}
                    {'🔥 Tertinggi!' if conv == cp['conversion_rate'].max() else ''}<br>
                    <b style="color:#9B72F5;">Rata-rata Usia:</b> {cp.loc[i,'avg_age']:.1f} tahun<br>
                    <b style="color:#9B72F5;">Rata-rata Durasi:</b> {cp.loc[i,'avg_duration']:.0f} detik<br>
                    <b style="color:#9B72F5;">Rata-rata Jumlah Kontak:</b> {cp.loc[i,'avg_campaign']:.1f}x<br>
                    <b style="color:#9B72F5;">Kontak Sebelumnya:</b> {cp.loc[i,'avg_previous']:.1f}x
                </div>""", unsafe_allow_html=True)

    with tab_c:
        insight(
            "Strategi segmentasi ini mengikuti framework <b>STP (Segmenting-Targeting-Positioning)</b> "
            "dalam konteks telemarketing. Setiap segmen memiliki karakteristik unik yang memerlukan "
            "pendekatan komunikasi dan alokasi sumber daya yang berbeda.",
            "FRAMEWORK STP"
        )
        for i in range(4):
            badge_class = ['cb-0', 'cb-1', 'cb-2', 'cb-3'][i]
            c1, c2 = st.columns([1, 4])
            with c1:
                st.markdown(f"""
                <div style="text-align:center; padding:1rem 0.5rem;
                    background:var(--bg-card); border:1px solid var(--border); border-radius:8px; margin-top:0.3rem;">
                    <div style="font-family:'Orbitron',monospace; font-size:1.8rem;
                                color:{CLUSTER_COLORS[i]}; font-weight:900;">C{i}</div>
                    <span class="cluster-badge {badge_class}">{cluster_names[i]}</span>
                </div>""", unsafe_allow_html=True)
            with c2:
                prio_text = {0: "RENDAH", 1: "SEDANG", 2: "TINGGI", 3: "PRIORITAS UTAMA"}[i]
                prio_color = {0: COLORS['red'], 1: COLORS['amber'], 2: COLORS['teal'], 3: COLORS['green']}[i]
                st.markdown(f"""
                <div class="insight-box" style="margin-top:0.3rem;">
                    <div class="ib-label">STRATEGI CLUSTER {i} —
                        <span style="color:{prio_color};">PRIORITAS: {prio_text}</span>
                    </div>
                    <div class="ib-text">{cluster_desc[i]}<br>
                        <span style="color:{prio_color}; font-weight:600;">{cluster_strategy[i]}</span>
                    </div>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: PREDIKSI RESPONS CAMPAIGN
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Prediksi Respons Campaign":
    page_header("🎯", "PREDIKSI RESPONS CAMPAIGN",
                "ML CLASSIFIER — FASE DESAIN & PILIHAN (SIMON, 1977)")

    test_res = models['test_results']

    # Model performance metrics
    st.markdown('<div class="sc-title">◈ PERFORMA MODEL — EVALUASI KLASIFIKASI</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Logistic Regression AUC", f"{test_res['lr_auc']:.4f}",
                  delta=f"+{(test_res['lr_auc']-0.5):.2%} vs baseline")
    with m2:
        st.metric("Random Forest AUC", f"{test_res['rf_auc']:.4f}",
                  delta=f"+{(test_res['rf_auc']-0.5):.2%} vs baseline")
    with m3:
        lr_acc = np.mean(test_res['lr_pred'] == test_res['y_test'])
        st.metric("LR Accuracy", f"{lr_acc:.1%}")
    with m4:
        rf_acc = np.mean(test_res['rf_pred'] == test_res['y_test'])
        st.metric("RF Accuracy", f"{rf_acc:.1%}")

    st.markdown('<div class="seg-divider"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.markdown('<div class="section-card"><div class="sc-title">◈ INPUT DATA NASABAH</div>', unsafe_allow_html=True)

        # Demographic
        st.markdown('<div style="font-family:\'Space Mono\',monospace; font-size:0.62rem; color:#6C3BDB; margin:0.5rem 0 0.3rem;">DEMOGRAFI</div>', unsafe_allow_html=True)
        age_in = st.slider("Usia", 18, 95, 40)
        job_in = st.selectbox("Pekerjaan", ['admin.', 'blue-collar', 'technician', 'services',
                                             'management', 'retired', 'entrepreneur',
                                             'self-employed', 'housemaid', 'unemployed', 'student', 'unknown'])
        marital_in = st.selectbox("Status Perkawinan", ['married', 'single', 'divorced', 'unknown'])
        education_in = st.selectbox("Pendidikan", ['university.degree', 'high.school', 'basic.9y',
                                                    'basic.4y', 'basic.6y', 'professional.course',
                                                    'illiterate', 'unknown'])

        st.markdown('<div style="font-family:\'Space Mono\',monospace; font-size:0.62rem; color:#00C8B4; margin:0.8rem 0 0.3rem;">FINANSIAL</div>', unsafe_allow_html=True)
        default_in = st.selectbox("Kredit Default", ['no', 'yes', 'unknown'])
        housing_in = st.selectbox("Pinjaman Rumah", ['no', 'yes', 'unknown'])
        loan_in = st.selectbox("Pinjaman Pribadi", ['no', 'yes', 'unknown'])

        st.markdown('<div style="font-family:\'Space Mono\',monospace; font-size:0.62rem; color:#F5A623; margin:0.8rem 0 0.3rem;">KAMPANYE</div>', unsafe_allow_html=True)
        contact_in = st.selectbox("Channel Kontak", ['cellular', 'telephone'])
        month_in = st.selectbox("Bulan", ['jan','feb','mar','apr','may','jun',
                                          'jul','aug','sep','oct','nov','dec'])
        day_in = st.selectbox("Hari", ['mon', 'tue', 'wed', 'thu', 'fri'])
        duration_in = st.slider("Estimasi Durasi (detik)", 0, 900, 250)
        campaign_in = st.slider("Jumlah Kontak Kampanye Ini", 1, 20, 2)
        pdays_in = st.slider("Hari Sejak Kontak Terakhir (999=belum pernah)", 0, 999, 999, step=1)
        previous_in = st.slider("Jumlah Kontak Sebelumnya", 0, 10, 0)
        poutcome_in = st.selectbox("Hasil Kampanye Sebelumnya", ['nonexistent', 'failure', 'success'])

        st.markdown('<div style="font-family:\'Space Mono\',monospace; font-size:0.62rem; color:#9B72F5; margin:0.8rem 0 0.3rem;">KONTEKS EKONOMI MAKRO</div>', unsafe_allow_html=True)
        emp_var = st.slider("Employment Variation Rate", -3.4, 1.4, -1.8, step=0.1)
        cons_price = st.slider("Consumer Price Index", 92.0, 95.0, 93.5, step=0.1)
        cons_conf = st.slider("Consumer Confidence Index", -51.0, -26.0, -40.0, step=0.5)
        euribor = st.slider("Euribor 3M Rate", 0.6, 5.0, 3.6, step=0.1)
        nr_emp = st.slider("Nr. Employed (ribuan)", 4963.0, 5228.0, 5099.0, step=10.0)

        predict_btn = st.button("⚡ ANALISIS PREDIKSI", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        if predict_btn or True:  # Show on load with defaults
            # Build input
            input_data = {
                'age': age_in, 'job': job_in, 'marital': marital_in,
                'education': education_in, 'default': default_in, 'housing': housing_in,
                'loan': loan_in, 'contact': contact_in, 'month': month_in,
                'day_of_week': day_in, 'duration': duration_in, 'campaign': campaign_in,
                'pdays': pdays_in, 'previous': previous_in, 'poutcome': poutcome_in,
                'emp.var.rate': emp_var, 'cons.price.idx': cons_price,
                'cons.conf.idx': cons_conf, 'euribor3m': euribor, 'nr.employed': nr_emp
            }
            input_df = pd.DataFrame([input_data])

            # Encode categoricals
            encoders = models['encoders']
            for col in ['job', 'marital', 'education', 'default', 'housing', 'loan',
                        'contact', 'month', 'day_of_week', 'poutcome']:
                le = encoders[col]
                val = input_df[col].values[0]
                if val in le.classes_:
                    input_df[col] = le.transform([val])[0]
                else:
                    input_df[col] = 0

            feature_cols = models['feature_cols']
            X_input = input_df[feature_cols]
            X_input_scaled = models['scaler'].transform(X_input)

            lr_proba = models['lr'].predict_proba(X_input_scaled)[0][1]
            rf_proba = models['rf'].predict_proba(X_input_scaled)[0][1]
            avg_proba = (lr_proba + rf_proba) / 2

            # Result display
            result_class = "pred-result-high" if avg_proba >= 0.5 else "pred-result-low"
            score_color = COLORS['green'] if avg_proba >= 0.5 else COLORS['red']
            verdict = "KEMUNGKINAN BERLANGGANAN" if avg_proba >= 0.5 else "KEMUNGKINAN TIDAK BERLANGGANAN"
            emoji = "✅" if avg_proba >= 0.5 else "❌"

            st.markdown(f"""
            <div class="{result_class}">
                <div class="pred-label" style="color:{score_color};">{emoji} PREDIKSI: {verdict}</div>
                <div class="pred-score" style="color:{score_color};">{avg_proba:.1%}</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.68rem; color:#8892B0; margin-top:0.5rem;">
                    Probabilitas Berlangganan Deposito Berjangka
                </div>
                <div style="display:flex; gap:1.5rem; justify-content:center; margin-top:0.8rem;">
                    <div style="font-size:0.78rem; color:#8892B0;">
                        LR: <span style="color:#9B72F5; font-weight:600;">{lr_proba:.1%}</span>
                    </div>
                    <div style="font-size:0.78rem; color:#8892B0;">
                        RF: <span style="color:#00C8B4; font-weight:600;">{rf_proba:.1%}</span>
                    </div>
                    <div style="font-size:0.78rem; color:#8892B0;">
                        Ensemble: <span style="color:{score_color}; font-weight:600;">{avg_proba:.1%}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=avg_proba * 100,
                number={'suffix': '%', 'font': {'size': 32, 'color': '#E8EAF6', 'family': 'Orbitron'}},
                delta={'reference': 50, 'suffix': '%',
                       'font': {'size': 14, 'color': '#8892B0'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1,
                             'tickcolor': '#546E7A', 'tickfont': {'color': '#546E7A', 'size': 10}},
                    'bar': {'color': score_color, 'thickness': 0.25},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(255,77,109,0.1)'},
                        {'range': [30, 50], 'color': 'rgba(245,166,35,0.1)'},
                        {'range': [50, 70], 'color': 'rgba(0,200,180,0.1)'},
                        {'range': [70, 100], 'color': 'rgba(46,204,113,0.15)'},
                    ],
                    'threshold': {
                        'line': {'color': COLORS['amber'], 'width': 2},
                        'thickness': 0.75,
                        'value': 50
                    }
                },
                title={'text': "SKOR PROBABILITAS KONVERSI",
                       'font': {'size': 11, 'color': '#8892B0', 'family': 'Space Mono'}}
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans, sans-serif', color='#8892B0', size=11),
                height=280, margin=dict(l=20, r=20, t=50, b=10),
                colorway=['#6C3BDB','#00C8B4','#F5A623','#FF4D6D','#2ECC71']
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Feature importance for this prediction
            fi = models['feature_importance']
            top_fi = fi.head(10)
            fig_fi = go.Figure(go.Bar(
                x=top_fi.values[::-1],
                y=top_fi.index[::-1],
                orientation='h',
                marker=dict(
                    color=top_fi.values[::-1],
                    colorscale=[[0, COLORS['purple']], [0.5, COLORS['teal']], [1, COLORS['green']]]
                )
            ))
            fig_fi.update_layout(**PLOTLY_LAYOUT,
                                 title='Top 10 Fitur Paling Berpengaruh (Random Forest)',
                                 height=320)
            st.plotly_chart(fig_fi, use_container_width=True)

            # Actionable insight
            if avg_proba >= 0.7:
                success_box(
                    f"Nasabah ini memiliki probabilitas tinggi ({avg_proba:.1%}). "
                    "Rekomendasikan penawaran deposito premium. Prioritaskan dalam antrian follow-up minggu ini.",
                    "REKOMENDASI TINDAKAN"
                )
            elif avg_proba >= 0.4:
                alert(
                    f"Probabilitas moderat ({avg_proba:.1%}). Pertimbangkan 1 kontak lanjutan "
                    "dengan skrip yang lebih personal. Evaluasi ulang setelah kontak berikutnya.",
                    "TINDAKAN YANG DISARANKAN"
                )
            else:
                st.markdown(f"""
                <div class="alert-box" style="border-left-color: #FF4D6D; background:rgba(255,77,109,0.06);">
                    <div class="ab-label" style="color:#FF4D6D;">⚠ RISIKO TINGGI</div>
                    <div class="ab-text">Probabilitas rendah ({avg_proba:.1%}). Hindari over-kontaking.
                    Pindahkan ke queue cold-outreach jangka panjang untuk efisiensi budget.</div>
                </div>""", unsafe_allow_html=True)

        # ROC Curve comparison
        y_test = test_res['y_test']
        from sklearn.metrics import roc_curve
        fpr_lr, tpr_lr, _ = roc_curve(y_test, test_res['lr_pred_proba'])
        fpr_rf, tpr_rf, _ = roc_curve(y_test, test_res['rf_pred_proba'])

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode='lines',
                                     name=f'Logistic Regression (AUC={test_res["lr_auc"]:.3f})',
                                     line=dict(color=COLORS['purple'], width=2)))
        fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines',
                                     name=f'Random Forest (AUC={test_res["rf_auc"]:.3f})',
                                     line=dict(color=COLORS['teal'], width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                     name='Random Baseline',
                                     line=dict(color=COLORS['muted'], width=1, dash='dash')))
        fig_roc.update_layout(**PLOTLY_LAYOUT,
                              title='ROC Curve — Perbandingan Model Klasifikasi',
                              xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate',
                              height=320)
        st.plotly_chart(fig_roc, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: ANALISIS EFEKTIVITAS CAMPAIGN
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Analisis Efektivitas Campaign":
    page_header("📊", "ANALISIS EFEKTIVITAS CAMPAIGN",
                "AGREGASI STATISTIK — FASE INTELIJEN & DESAIN (SIMON, 1977)")

    total_contacts = len(df)
    total_conv = df['y'].sum()
    overall_cr = df['y'].mean()
    avg_campaign_contacts = df['campaign'].mean()

    st.markdown(f"""
    <div class="kpi-grid">
        {render_kpi("TOTAL KONTAK", f"{total_contacts:,}", "seluruh kampanye", variant="kpi-teal")}
        {render_kpi("TOTAL KONVERSI", f"{int(total_conv):,}", "nasabah terkonversi", variant="kpi-green")}
        {render_kpi("OVERALL CONV RATE", f"{overall_cr:.1%}", "tingkat keberhasilan")}
        {render_kpi("RATA2 KONTAK/NASABAH", f"{avg_campaign_contacts:.1f}x", "per kampanye", variant="kpi-amber")}
        {render_kpi("BEST CHANNEL", "Cellular", f"{df[df['contact']=='cellular']['y'].mean():.1%} conv rate", variant="kpi-teal")}
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📅 TEMPORAL", "📞 CHANNEL & FREKUENSI", "🔄 POUTCOME", "🌍 KONTEKS EKONOMI"])

    with tab1:
        month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
        month_data = df.groupby('month').agg(
            contacts=('y', 'count'),
            conversions=('y', 'sum'),
            conv_rate=('y', 'mean'),
            avg_duration=('duration', 'mean')
        ).reindex(month_order).dropna().reset_index()

        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=['Volume Kontak per Bulan', 'Conversion Rate per Bulan',
                                            'Rata-rata Durasi per Bulan', 'Kontak vs Konversi (Bubble)'])
        # Volume
        fig.add_trace(go.Bar(x=month_data['month'], y=month_data['contacts'],
                             marker_color=COLORS['purple'], name='Volume',
                             showlegend=False), row=1, col=1)
        # Conv rate
        fig.add_trace(go.Scatter(x=month_data['month'], y=month_data['conv_rate'] * 100,
                                 mode='lines+markers',
                                 line=dict(color=COLORS['teal'], width=2.5),
                                 marker=dict(size=8, color=COLORS['teal'],
                                             line=dict(color='white', width=1.5)),
                                 name='Conv Rate %', showlegend=False), row=1, col=2)
        # Avg duration
        fig.add_trace(go.Bar(x=month_data['month'], y=month_data['avg_duration'],
                             marker_color=COLORS['amber'], name='Durasi',
                             showlegend=False), row=2, col=1)
        # Bubble
        fig.add_trace(go.Scatter(
            x=month_data['contacts'], y=month_data['conv_rate'] * 100,
            mode='markers+text', text=month_data['month'],
            textposition='top center',
            marker=dict(size=month_data['conversions'] / 5,
                        color=month_data['conv_rate'],
                        colorscale=[[0, COLORS['purple']], [1, COLORS['green']]],
                        showscale=False),
            showlegend=False
        ), row=2, col=2)

        fig.update_layout(**PLOTLY_LAYOUT, height=550,
                          title='Dashboard Temporal Efektivitas Kampanye')
        fig.update_annotations(font=dict(color='#8892B0', size=10))
        st.plotly_chart(fig, use_container_width=True)

        # Best months
        top3_months = month_data.nlargest(3, 'conv_rate')
        best = ', '.join([f"<b style='color:{COLORS['teal']};'>{m.upper()}</b>"
                         for m in top3_months['month'].values])
        insight(f"Bulan dengan conversion rate tertinggi: {best}. "
                "Bulan-bulan ini cenderung bertepatan dengan kondisi ekonomi yang lebih stabil. "
                "Alokasikan lebih banyak agen dan anggaran kontak di bulan-bulan ini.",
                "TEMPORAL INSIGHT")

        # Day of week analysis
        day_order = ['mon', 'tue', 'wed', 'thu', 'fri']
        day_data = df.groupby('day_of_week').agg(
            conv_rate=('y', 'mean'), count=('y', 'count')
        ).reindex(day_order).reset_index()

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=day_data['day_of_week'], y=day_data['conv_rate'] * 100,
            marker=dict(color=day_data['conv_rate'] * 100,
                        colorscale=[[0, COLORS['purple']], [1, COLORS['teal']]]),
            text=[f"{v:.1f}%" for v in day_data['conv_rate'] * 100],
            textposition='outside', textfont=dict(color='#E8EAF6', size=11)
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, title='Conversion Rate per Hari dalam Seminggu',
                           yaxis_ticksuffix='%', height=280)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            # Channel comparison
            ch_data = df.groupby('contact').agg(
                conv_rate=('y', 'mean'), count=('y', 'count')
            ).reset_index()
            fig = go.Figure()
            for _, row in ch_data.iterrows():
                color = COLORS['teal'] if row['contact'] == 'cellular' else COLORS['purple']
                fig.add_trace(go.Bar(
                    x=[row['contact'].upper()],
                    y=[row['conv_rate'] * 100],
                    name=row['contact'],
                    marker_color=color,
                    text=[f"{row['conv_rate']:.1%}<br>n={row['count']:,}"],
                    textposition='outside',
                    textfont=dict(color='#E8EAF6')
                ))
            fig.update_layout(**PLOTLY_LAYOUT, title='Efektivitas per Channel Kontak',
                              yaxis_ticksuffix='%', showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Campaign frequency vs conversion
            camp_data = df.copy()
            camp_data['camp_bin'] = pd.cut(camp_data['campaign'], bins=[0, 1, 2, 3, 5, 56],
                                           labels=['1x', '2x', '3x', '4-5x', '>5x'])
            camp_conv = camp_data.groupby('camp_bin', observed=True).agg(
                conv_rate=('y', 'mean'), count=('y', 'count')
            ).reset_index()

            fig = go.Figure()
            colors_bar = [COLORS['green'], COLORS['teal'], COLORS['amber'], COLORS['red'], '#800020']
            for i, (_, row) in enumerate(camp_conv.iterrows()):
                fig.add_trace(go.Bar(
                    x=[str(row['camp_bin'])], y=[row['conv_rate'] * 100],
                    name=str(row['camp_bin']),
                    marker_color=colors_bar[i % len(colors_bar)],
                    text=[f"{row['conv_rate']:.1%}\nn={row['count']:,}"],
                    textposition='outside',
                    textfont=dict(color='#E8EAF6', size=10),
                    showlegend=False
                ))
            fig.update_layout(**PLOTLY_LAYOUT, title='Conversion Rate vs Frekuensi Kontak',
                              xaxis_title='Jumlah Kontak Kampanye',
                              yaxis_ticksuffix='%', height=320)
            st.plotly_chart(fig, use_container_width=True)

        alert(
            "Data menunjukkan pola <b>diminishing returns</b> — semakin banyak kontak, semakin turun "
            "conversion rate. Nasabah yang dihubungi lebih dari 5x memiliki tingkat konversi paling rendah. "
            "Tetapkan batas maksimum <b>3x kontak per nasabah</b> per kampanye untuk efisiensi optimal.",
            "POLA DIMINISHING RETURNS"
        )

    with tab3:
        pout_data = df.groupby('poutcome').agg(
            conv_rate=('y', 'mean'), count=('y', 'count')
        ).reset_index()

        col1, col2 = st.columns(2)
        with col1:
            pout_colors = {'success': COLORS['green'], 'failure': COLORS['red'],
                           'nonexistent': COLORS['muted']}
            fig = go.Figure(go.Bar(
                x=pout_data['poutcome'].str.upper(),
                y=pout_data['conv_rate'] * 100,
                marker_color=[pout_colors.get(p, COLORS['purple']) for p in pout_data['poutcome']],
                text=[f"{v:.1%}\nn={n:,}" for v, n in zip(pout_data['conv_rate'], pout_data['count'])],
                textposition='outside', textfont=dict(color='#E8EAF6', size=10)
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title='Conversion Rate berdasarkan Hasil Kampanye Sebelumnya',
                              yaxis_ticksuffix='%', height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Previous contacts analysis
            prev_data = df[df['previous'] <= 6].groupby('previous').agg(
                conv_rate=('y', 'mean'), count=('y', 'count')
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=prev_data['previous'], y=prev_data['conv_rate'] * 100,
                mode='lines+markers',
                line=dict(color=COLORS['teal'], width=2.5),
                marker=dict(size=prev_data['count'] / 30,
                            color=COLORS['purple'],
                            line=dict(color=COLORS['teal'], width=1.5)),
                text=[f"n={n:,}" for n in prev_data['count']],
                hovertemplate='%{x}x kontak<br>Conv: %{y:.1f}%<br>%{text}'
            ))
            fig.update_layout(**PLOTLY_LAYOUT,
                              title='Conv Rate vs Jumlah Kontak Sebelumnya',
                              xaxis_title='Jumlah Kontak Sebelumnya',
                              yaxis_ticksuffix='%', height=350)
            st.plotly_chart(fig, use_container_width=True)

        success_box(
            "Nasabah yang sebelumnya sukses dikonversi memiliki kemungkinan <b>3-5x lebih tinggi</b> "
            "untuk berlangganan lagi. Prioritaskan database nasabah lama dengan status 'success' "
            "sebagai target utama kampanye baru.",
            "STRATEGI RETARGETING"
        )

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            # Euribor vs conversion scatter
            df_sample = df.sample(500, random_state=42)
            fig = go.Figure()
            for target_val, color, name in [(1, COLORS['green'], 'Berlangganan'), (0, COLORS['purple'], 'Tidak')]:
                mask = df_sample['y'] == target_val
                fig.add_trace(go.Scatter(
                    x=df_sample[mask]['euribor3m'],
                    y=df_sample[mask]['cons.conf.idx'],
                    mode='markers', name=name,
                    marker=dict(color=color, size=5, opacity=0.6)
                ))
            fig.update_layout(**PLOTLY_LAYOUT,
                              title='Euribor vs Consumer Confidence (Colored by Target)',
                              xaxis_title='Euribor 3M Rate',
                              yaxis_title='Consumer Confidence Index',
                              height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Employment rate bins vs conversion
            df['emp_bin'] = pd.cut(df['emp.var.rate'], bins=5)
            emp_data = df.groupby('emp_bin', observed=True).agg(
                conv_rate=('y', 'mean'), count=('y', 'count')
            ).reset_index()
            emp_data['emp_bin'] = emp_data['emp_bin'].astype(str)
            fig = go.Figure(go.Bar(
                x=emp_data['emp_bin'], y=emp_data['conv_rate'] * 100,
                marker=dict(color=emp_data['conv_rate'] * 100,
                            colorscale=[[0, COLORS['red']], [0.5, COLORS['amber']], [1, COLORS['green']]]),
                text=[f"{v:.1f}%" for v in emp_data['conv_rate'] * 100],
                textposition='outside', textfont=dict(color='#E8EAF6', size=10)
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title='Conv Rate vs Employment Variation Rate',
                              xaxis_title='Employment Variation Rate (bin)',
                              yaxis_ticksuffix='%', height=350)
            fig.update_xaxes(tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)

        insight(
            "Kondisi ekonomi makro memiliki pengaruh signifikan terhadap keputusan nasabah. "
            "Tingkat euribor rendah dan consumer confidence yang lebih tinggi (mendekati nol) "
            "berkorelasi dengan conversion rate yang lebih baik. "
            "Jadwalkan kampanye intensif saat indikator ekonomi menunjukkan stabilitas.",
            "INSIGHT EKONOMI MAKRO"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: SIMULASI & OPTIMASI BUDGET
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💰 Simulasi & Optimasi Budget":
    page_header("💰", "SIMULASI & OPTIMASI BUDGET",
                "MONTE CARLO SIMULATION — FASE PILIHAN (SIMON, 1977)")

    alert(
        "Dataset UCI tidak mencakup data biaya per kontak. Sistem ini menggunakan <b>asumsi biaya yang dapat dikonfigurasi</b> "
        "dan metode simulasi Monte Carlo untuk menghasilkan distribusi ROI yang realistis. "
        "Sesuaikan parameter sesuai kondisi aktual bank Anda.",
        "ASUMSI MODEL"
    )

    col_param, col_result = st.columns([2, 3])

    with col_param:
        st.markdown('<div class="section-card"><div class="sc-title">◈ PARAMETER SIMULASI</div>', unsafe_allow_html=True)

        st.markdown('<div style="font-family:\'Space Mono\',monospace; font-size:0.62rem; color:#6C3BDB; margin:0.5rem 0 0.3rem;">BIAYA & ANGGARAN</div>', unsafe_allow_html=True)
        budget_total = st.number_input("Total Budget Kampanye (Rp)", min_value=1_000_000,
                                       max_value=1_000_000_000, value=50_000_000, step=1_000_000,
                                       format="%d")
        cost_per_contact = st.slider("Biaya per Kontak (Rp)", 1_000, 50_000, 5_000, step=500)
        deposit_value = st.slider("Nilai Deposito per Nasabah (Rp)", 1_000_000, 50_000_000,
                                  10_000_000, step=500_000)
        bank_margin = st.slider("Margin Bank dari Deposito (%)", 0.5, 5.0, 1.5, step=0.1)

        st.markdown('<div style="font-family:\'Space Mono\',monospace; font-size:0.62rem; color:#00C8B4; margin:0.8rem 0 0.3rem;">PARAMETER PROBABILISTIK</div>', unsafe_allow_html=True)
        base_cr = st.slider("Base Conversion Rate (%)", 5.0, 50.0,
                            float(df['y'].mean() * 100), step=0.5)
        cr_std = st.slider("Volatilitas Conversion Rate (std %)", 0.5, 10.0, 3.0, step=0.5)
        cost_uncertainty = st.slider("Ketidakpastian Biaya (±%)", 0.0, 30.0, 10.0, step=1.0)

        st.markdown('<div style="font-family:\'Space Mono\',monospace; font-size:0.62rem; color:#F5A623; margin:0.8rem 0 0.3rem;">SKENARIO SEGMEN TARGET</div>', unsafe_allow_html=True)
        segment_choice = st.selectbox("Target Segmen", [
            "Semua Nasabah (Overall)",
            "C0: Skeptics (+0%)",
            "C1: Loyalists (+2%)",
            "C2: Prospects (+3%)",
            "C3: Champions (+5%)",
        ])
        seg_boost = {'Semua Nasabah (Overall)': 0, 'C0: Skeptics (+0%)': 0,
                     'C1: Loyalists (+2%)': 2, 'C2: Prospects (+3%)': 3,
                     'C3: Champions (+5%)': 5}
        boost = seg_boost.get(segment_choice, 0)

        n_simulations = st.slider("Jumlah Simulasi Monte Carlo", 500, 5000, 2000, step=500)

        run_sim = st.button("🎲 JALANKAN SIMULASI MONTE CARLO", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if run_sim or 'mc_results' in st.session_state:
            if run_sim:
                # Run Monte Carlo
                np.random.seed(None)
                effective_cr = (base_cr + boost) / 100
                max_contacts = int(budget_total / cost_per_contact)
                annual_margin = deposit_value * (bank_margin / 100) * 12

                sim_revenues = []
                sim_costs = []
                sim_rois = []
                sim_conversions = []

                for _ in range(n_simulations):
                    sim_cr = np.clip(np.random.normal(effective_cr, cr_std / 100), 0.01, 0.99)
                    cost_multiplier = 1 + np.random.uniform(-cost_uncertainty/100, cost_uncertainty/100)
                    actual_cost_per_contact = cost_per_contact * cost_multiplier
                    actual_contacts = int(budget_total / actual_cost_per_contact)
                    actual_contacts = min(actual_contacts, max_contacts)

                    conversions = int(actual_contacts * sim_cr)
                    total_cost = actual_contacts * actual_cost_per_contact
                    total_revenue = conversions * annual_margin
                    roi = (total_revenue - total_cost) / total_cost * 100 if total_cost > 0 else 0

                    sim_revenues.append(total_revenue)
                    sim_costs.append(total_cost)
                    sim_rois.append(roi)
                    sim_conversions.append(conversions)

                st.session_state['mc_results'] = {
                    'revenues': sim_revenues, 'costs': sim_costs,
                    'rois': sim_rois, 'conversions': sim_conversions,
                    'max_contacts': max_contacts, 'effective_cr': effective_cr,
                    'annual_margin': annual_margin
                }

            mc = st.session_state['mc_results']
            rois = np.array(mc['rois'])
            revenues = np.array(mc['revenues'])
            convs = np.array(mc['conversions'])

            # KPIs
            st.markdown(f"""
            <div class="kpi-grid">
                {render_kpi("MEDIAN ROI", f"{np.median(rois):.1f}%", "50th percentile", variant="kpi-teal")}
                {render_kpi("PROB ROI POSITIF", f"{np.mean(rois>0):.1%}", "dari simulasi", variant="kpi-green")}
                {render_kpi("MEDIAN KONVERSI", f"{int(np.median(convs)):,}", "nasabah", variant="kpi-amber")}
                {render_kpi("MAX KONTAK", f"{mc['max_contacts']:,}", f"@ Rp{cost_per_contact:,}/kontak")}
            </div>""", unsafe_allow_html=True)

            # ROI Distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=rois, nbinsx=60,
                marker=dict(color=COLORS['purple'], opacity=0.7,
                            line=dict(color=COLORS['purple_light'], width=0.3)),
                name='Distribusi ROI'
            ))
            # Add percentile lines
            for pct, color, label in [(5, COLORS['red'], 'P5'),
                                       (25, COLORS['amber'], 'P25'),
                                       (50, COLORS['teal'], 'P50'),
                                       (75, COLORS['green'], 'P75'),
                                       (95, '#00FF88', 'P95')]:
                val = np.percentile(rois, pct)
                fig.add_vline(x=val, line_dash='dash', line_color=color,
                              annotation_text=f"{label}: {val:.0f}%",
                              annotation_font_color=color, annotation_font_size=9)
            fig.add_vline(x=0, line_color='white', line_width=1.5,
                          annotation_text='BREAKEVEN', annotation_font_color='white')
            fig.update_layout(**PLOTLY_LAYOUT,
                              title=f'Distribusi ROI — {n_simulations:,} Simulasi Monte Carlo',
                              xaxis_title='Return on Investment (%)',
                              yaxis_title='Frekuensi', height=320,
                              xaxis_ticksuffix='%')
            st.plotly_chart(fig, use_container_width=True)

            # Revenue vs Cost scatter
            col1, col2 = st.columns(2)
            with col1:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=np.array(mc['costs']) / 1_000_000,
                    y=np.array(revenues) / 1_000_000,
                    mode='markers',
                    marker=dict(color=rois, colorscale=[[0, COLORS['red']], [0.5, COLORS['amber']], [1, COLORS['green']]],
                                size=3, opacity=0.4,
                                colorbar=dict(title='ROI%', tickfont=dict(color='#8892B0', size=8))),
                ))
                fig2.add_trace(go.Scatter(
                    x=[min(mc['costs'])/1e6, max(mc['costs'])/1e6],
                    y=[min(mc['costs'])/1e6, max(mc['costs'])/1e6],
                    mode='lines', name='Breakeven',
                    line=dict(color='white', dash='dash', width=1)
                ))
                fig2.update_layout(**PLOTLY_LAYOUT, title='Revenue vs Cost per Simulasi',
                                   xaxis_title='Total Cost (Juta Rp)',
                                   yaxis_title='Total Revenue (Juta Rp)', height=280)
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                # Conversion distribution
                fig3 = go.Figure(go.Histogram(
                    x=convs, nbinsx=40,
                    marker=dict(color=COLORS['teal'], opacity=0.7)
                ))
                fig3.add_vline(x=np.median(convs), line_dash='dash', line_color=COLORS['amber'],
                               annotation_text=f"Median: {int(np.median(convs)):,}",
                               annotation_font_color=COLORS['amber'])
                fig3.update_layout(**PLOTLY_LAYOUT, title='Distribusi Konversi yang Diharapkan',
                                   xaxis_title='Jumlah Nasabah Terkonversi',
                                   yaxis_title='Frekuensi', height=280)
                st.plotly_chart(fig3, use_container_width=True)

            # Risk summary table
            st.markdown('<div class="sc-title">◈ RINGKASAN RISIKO (VALUE AT RISK)</div>', unsafe_allow_html=True)
            risk_df = pd.DataFrame({
                'Skenario': ['Pesimis (P5)', 'Konservatif (P25)', 'Base Case (P50)',
                             'Optimis (P75)', 'Terbaik (P95)'],
                'ROI (%)': [f"{np.percentile(rois,p):.1f}%" for p in [5, 25, 50, 75, 95]],
                'Revenue (Rp Juta)': [f"Rp {np.percentile(revenues,p)/1e6:.1f}M" for p in [5, 25, 50, 75, 95]],
                'Konversi': [f"{int(np.percentile(convs,p)):,}" for p in [5, 25, 50, 75, 95]],
                'Status': ['🔴 Rugi' if np.percentile(rois,p) < 0 else '🟡 Tipis' if np.percentile(rois,p) < 20
                           else '🟢 Positif' for p in [5, 25, 50, 75, 95]]
            })
            st.dataframe(risk_df, use_container_width=True, hide_index=True)

            prob_pos = np.mean(rois > 0)
            if prob_pos > 0.8:
                success_box(
                    f"Dengan parameter yang dimasukkan, probabilitas ROI positif adalah <b>{prob_pos:.1%}</b>. "
                    f"Skenario ini layak dieksekusi. Median ROI <b>{np.median(rois):.1f}%</b> "
                    f"dengan target konversi <b>{int(np.median(convs)):,} nasabah</b>.",
                    "KEPUTUSAN: DIREKOMENDASIKAN"
                )
            elif prob_pos > 0.5:
                alert(
                    f"Probabilitas ROI positif: <b>{prob_pos:.1%}</b>. Risiko moderat. "
                    "Pertimbangkan mengurangi biaya per kontak atau menargetkan segmen dengan conversion rate lebih tinggi.",
                    "KEPUTUSAN: PERLU OPTIMASI"
                )
            else:
                st.markdown(f"""
                <div class="alert-box" style="border-left-color:#FF4D6D; background:rgba(255,77,109,0.06);">
                    <div class="ab-label" style="color:#FF4D6D;">⚠ KEPUTUSAN: RISIKO TINGGI</div>
                    <div class="ab-text">Probabilitas ROI positif hanya <b>{prob_pos:.1%}</b>.
                    Revisi strategi sebelum eksekusi. Pertimbangkan scaling down budget
                    atau menargetkan segmen Champions (C3) secara eksklusif.</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                        height:400px; text-align:center; opacity:0.6;">
                <div style="font-size:3rem; margin-bottom:1rem;">🎲</div>
                <div style="font-family:'Orbitron',monospace; font-size:1rem; color:#8892B0; letter-spacing:0.1em;">
                    KLIK "JALANKAN SIMULASI"
                </div>
                <div style="font-size:0.78rem; color:#546E7A; margin-top:0.5rem;">
                    untuk memulai simulasi Monte Carlo
                </div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: REKOMENDASI KEPUTUSAN
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧠 Rekomendasi Keputusan":
    page_header("🧠", "REKOMENDASI KEPUTUSAN",
                "INTEGRATED DECISION ENGINE — FASE PILIHAN (SIMON, 1977)")

    insight(
        "Halaman ini mengintegrasikan seluruh output dari 5 modul DSS — segmentasi, prediksi, "
        "analisis efektivitas, dan simulasi budget — menjadi satu kerangka rekomendasi yang dapat dieksekusi. "
        "Berdasarkan <b>Model Pengambilan Keputusan Simon (1977): Intelligence → Design → Choice</b>.",
        "INTEGRATED DSS OUTPUT"
    )

    st.markdown('<div class="seg-divider"></div>', unsafe_allow_html=True)

    # ── Intelligence Phase Summary ──
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.75rem; color:#6C3BDB;
                letter-spacing:0.15em; margin-bottom:1rem;">
        ◈ FASE 1: INTELLIGENCE — KONDISI AKTUAL
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="section-card">
            <div class="sc-title">PROFIL DATASET</div>
            <div style="font-size:0.85rem; color:#E8EAF6; line-height:2;">
                📊 Total Nasabah: <b style="color:#00C8B4;">{len(df):,}</b><br>
                ✅ Conversion Rate: <b style="color:#2ECC71;">{df['y'].mean():.1%}</b><br>
                📞 Rata2 Kontak: <b style="color:#F5A623;">{df['campaign'].mean():.1f}x</b><br>
                ⏱ Durasi Optimal: <b style="color:#9B72F5;">&gt;180 detik</b>
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        best_month = df.groupby('month')['y'].mean().idxmax().upper()
        best_channel = df.groupby('contact')['y'].mean().idxmax().upper()
        st.markdown(f"""
        <div class="section-card">
            <div class="sc-title">KONDISI TERBAIK</div>
            <div style="font-size:0.85rem; color:#E8EAF6; line-height:2;">
                📅 Bulan Terbaik: <b style="color:#00C8B4;">{best_month}</b><br>
                📱 Channel Terbaik: <b style="color:#2ECC71;">{best_channel}</b><br>
                🔄 Status Sebelumnya: <b style="color:#F5A623;">SUCCESS</b><br>
                🎯 Segmen Prioritas: <b style="color:#9B72F5;">Champions (C3)</b>
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        tr = models['test_results']
        st.markdown(f"""
        <div class="section-card">
            <div class="sc-title">PERFORMA MODEL ML</div>
            <div style="font-size:0.85rem; color:#E8EAF6; line-height:2;">
                🤖 LR AUC: <b style="color:#00C8B4;">{tr['lr_auc']:.4f}</b><br>
                🌲 RF AUC: <b style="color:#2ECC71;">{tr['rf_auc']:.4f}</b><br>
                🎯 Ensemble: <b style="color:#F5A623;">{(tr['lr_auc']+tr['rf_auc'])/2:.4f}</b><br>
                📌 Threshold: <b style="color:#9B72F5;">50% (binary)</b>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="seg-divider"></div>', unsafe_allow_html=True)

    # ── Design Phase ──
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.75rem; color:#00C8B4;
                letter-spacing:0.15em; margin-bottom:1rem;">
        ◈ FASE 2: DESIGN — ALTERNATIF KEPUTUSAN
    </div>""", unsafe_allow_html=True)

    tab_a, tab_b, tab_c = st.tabs(["🎯 MATRIKS PRIORITAS SEGMEN", "📋 ACTION PLAN", "⚖️ EVALUASI KRITERIA"])

    with tab_a:
        cp = models['cluster_profiles']
        # Priority scoring matrix
        seg_matrix = []
        for i in range(4):
            conv = cp.loc[i, 'conversion_rate']
            size = cp.loc[i, 'size']
            prev = cp.loc[i, 'avg_previous']
            # Priority score: weighted composite
            priority = (conv * 0.5 + (size/cp['size'].max()) * 0.2 +
                       (prev/cp['avg_previous'].max() if cp['avg_previous'].max() > 0 else 0) * 0.3)
            seg_matrix.append({
                'Cluster': f"C{i}",
                'Nama Segmen': {0: 'Skeptics', 1: 'Loyalists', 2: 'Prospects', 3: 'Champions'}[i],
                'Ukuran': cp.loc[i, 'size'],
                'Conv Rate': conv,
                'Priority Score': priority,
                'Budget Share': None  # Will assign after
            })
        seg_matrix = sorted(seg_matrix, key=lambda x: x['Priority Score'], reverse=True)
        # Assign budget proportional to priority
        total_score = sum(s['Priority Score'] for s in seg_matrix)
        for s in seg_matrix:
            s['Budget Share'] = s['Priority Score'] / total_score

        # Bubble chart: size = budget allocation
        fig = go.Figure()
        cluster_colors_all = [COLORS['purple'], COLORS['teal'], COLORS['amber'], COLORS['green']]
        for s in seg_matrix:
            ci = int(s['Cluster'][1])
            fig.add_trace(go.Scatter(
                x=[s['Ukuran']], y=[s['Conv Rate'] * 100],
                mode='markers+text',
                text=[f"C{ci}: {s['Nama Segmen']}<br>Budget: {s['Budget Share']:.1%}"],
                textposition='top center',
                textfont=dict(size=10, color='#E8EAF6'),
                marker=dict(size=s['Budget Share'] * 400,
                            color=cluster_colors_all[ci],
                            opacity=0.8,
                            line=dict(color='white', width=1.5)),
                name=f"C{ci}: {s['Nama Segmen']}",
                hovertemplate=(f"<b>C{ci}: {s['Nama Segmen']}</b><br>"
                               f"Ukuran: {s['Ukuran']:,}<br>"
                               f"Conv Rate: {s['Conv Rate']:.1%}<br>"
                               f"Priority Score: {s['Priority Score']:.3f}<br>"
                               f"Budget Allocation: {s['Budget Share']:.1%}")
            ))
        fig.update_layout(**PLOTLY_LAYOUT,
                          title='Matriks Prioritas Segmen (bubble size = alokasi budget)',
                          xaxis_title='Ukuran Segmen (nasabah)',
                          yaxis_title='Conversion Rate (%)',
                          yaxis_ticksuffix='%', height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Priority table
        prio_df = pd.DataFrame([{
            'Rank': i+1,
            'Cluster': s['Cluster'],
            'Segmen': s['Nama Segmen'],
            'Ukuran': f"{s['Ukuran']:,}",
            'Conv Rate': f"{s['Conv Rate']:.1%}",
            'Priority Score': f"{s['Priority Score']:.3f}",
            'Alokasi Budget': f"{s['Budget Share']:.1%}",
        } for i, s in enumerate(seg_matrix)])
        st.dataframe(prio_df, use_container_width=True, hide_index=True)

    with tab_b:
        st.markdown('<div class="sc-title">◈ INTEGRATED ACTION PLAN — 3 HORIZON</div>', unsafe_allow_html=True)

        horizons = [
            {
                'horizon': 'JANGKA PENDEK (0-30 Hari)',
                'color': COLORS['green'],
                'icon': '⚡',
                'actions': [
                    ('Fokus kontak Champions (C3)', 'Alokasi 45% budget dan agen terbaik untuk segmen ini'),
                    ('Gunakan channel Cellular', f"Conversion rate {df[df['contact']=='cellular']['y'].mean():.1%} vs telephone {df[df['contact']=='telephone']['y'].mean():.1%}"),
                    ('Batasi kontak maks 3x/nasabah', 'Hindari diminishing returns & persepsi negatif'),
                    ('Prioritas waktu: pagi hari', 'Hari Kamis & Selasa menunjukkan conversion rate tertinggi'),
                ]
            },
            {
                'horizon': 'JANGKA MENENGAH (1-3 Bulan)',
                'color': COLORS['amber'],
                'icon': '📈',
                'actions': [
                    ('Retarget database success lama', 'Nasabah dengan poutcome=success punya conv rate 3-5x lebih tinggi'),
                    ('Optimalkan skrip berdasarkan model', f"Gunakan threshold probabilitas ≥50% sebagai filter prioritas kontak"),
                    ('Kalibrasi budget via Monte Carlo', 'Jalankan simulasi tiap kuartal dengan data aktual terbaru'),
                    ('Tingkatkan Prospects (C2)', 'Potential conversion tertinggi untuk investasi jangka menengah'),
                ]
            },
            {
                'horizon': 'JANGKA PANJANG (3-12 Bulan)',
                'color': COLORS['purple_light'],
                'icon': '🚀',
                'actions': [
                    ('Retrain model dengan data aktual', 'Replace synthetic data dengan bank-additional-full.csv asli'),
                    ('Integrasikan data biaya real', 'CRM integration untuk cost tracking aktual per kontak'),
                    ('Feature engineering lanjutan', 'Tambahkan variabel musiman, event ekonomi, suku bunga BI'),
                    ('A/B testing skrip kampanye', 'Gunakan model prediksi untuk assignment kelompok eksperimen'),
                ]
            }
        ]

        for h in horizons:
            st.markdown(f"""
            <div class="section-card" style="border-left: 3px solid {h['color']};">
                <div class="sc-title" style="color:{h['color']};">{h['icon']} {h['horizon']}</div>
            """, unsafe_allow_html=True)
            for action, detail in h['actions']:
                st.markdown(f"""
                <div style="display:flex; gap:0.8rem; margin-bottom:0.6rem; align-items:flex-start;">
                    <div style="color:{h['color']}; font-size:0.9rem; margin-top:0.1rem; flex-shrink:0;">▸</div>
                    <div>
                        <div style="color:#E8EAF6; font-size:0.88rem; font-weight:500;">{action}</div>
                        <div style="color:#8892B0; font-size:0.78rem; margin-top:0.1rem;">{detail}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_c:
        st.markdown('<div class="sc-title">◈ EVALUASI MULTI-KRITERIA (MCDM)</div>', unsafe_allow_html=True)

        insight(
            "Metode evaluasi multi-kriteria (MCDM) mempertimbangkan 5 dimensi keputusan secara simultan: "
            "profitabilitas, efisiensi operasional, risiko eksekusi, dampak jangka panjang, dan kelayakan teknis. "
            "Setiap alternatif strategi dievaluasi terhadap bobot kepentingan yang ditetapkan manajemen.",
            "METODOLOGI MCDM"
        )

        # Define alternatives & criteria
        alternatives = ['Strategi A\n(Mass Contact)', 'Strategi B\n(Segmented)', 'Strategi C\n(Champions Only)', 'Strategi D\n(ML-Driven)']
        criteria = ['Profitabilitas', 'Efisiensi Biaya', 'Risiko Rendah', 'Skalabilitas', 'Kecepatan ROI']
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]

        raw_scores = np.array([
            [0.55, 0.40, 0.70, 0.80, 0.60],  # Mass Contact
            [0.75, 0.70, 0.65, 0.75, 0.70],  # Segmented
            [0.85, 0.80, 0.55, 0.50, 0.85],  # Champions Only
            [0.80, 0.75, 0.70, 0.85, 0.75],  # ML-Driven
        ])

        weighted_scores = raw_scores * weights
        total_scores = weighted_scores.sum(axis=1)

        # Heatmap
        fig = go.Figure(go.Heatmap(
            z=raw_scores,
            x=criteria, y=alternatives,
            colorscale=[[0, COLORS['red']], [0.5, COLORS['amber']], [1, COLORS['green']]],
            text=np.round(raw_scores, 2), texttemplate='%{text}',
            textfont=dict(size=12, color='white'),
            hovertemplate='%{y}<br>%{x}: %{z:.2f}<extra></extra>',
            zmid=0.6
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title='Matriks Skor MCDM (0 = buruk, 1 = terbaik)',
                          height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Final ranking
        rank_df = pd.DataFrame({
            'Rank': range(1, 5),
            'Strategi': [alternatives[i].replace('\n', ' ') for i in np.argsort(-total_scores)],
            'Total Score': [f"{total_scores[i]:.4f}" for i in np.argsort(-total_scores)],
            'Rekomendasi': ['✅ TERPILIH', '🟡 Alternatif', '🟡 Alternatif', '🔴 Tidak Disarankan']
        })
        st.dataframe(rank_df, use_container_width=True, hide_index=True)

        # Winner
        winner_idx = np.argmax(total_scores)
        winner = alternatives[winner_idx].replace('\n', ' ')
        success_box(
            f"Berdasarkan analisis MCDM dengan 5 kriteria dan pembobotan manajemen, <b>{winner}</b> "
            f"adalah strategi yang paling direkomendasikan dengan skor tertinggi <b>{total_scores[winner_idx]:.4f}</b>. "
            "Strategi ini menyeimbangkan profitabilitas, efisiensi biaya, dan skalabilitas secara optimal.",
            "KEPUTUSAN FINAL DSS"
        )

    st.markdown('<div class="seg-divider"></div>', unsafe_allow_html=True)

    # ── Choice Phase ──
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.75rem; color:#F5A623;
                letter-spacing:0.15em; margin-bottom:1rem;">
        ◈ FASE 3: CHOICE — RINGKASAN KEPUTUSAN EKSEKUTIF
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(108,59,219,0.12), rgba(0,200,180,0.08));
                border: 1px solid rgba(108,59,219,0.3); border-radius:12px; padding:1.8rem 2rem;">
        <div style="font-family:'Orbitron',monospace; font-size:1rem; color:#E8EAF6;
                    font-weight:700; letter-spacing:0.1em; margin-bottom:1.2rem;">
            🏦 REKOMENDASI KEPUTUSAN AKHIR — BANK MARKETING DSS
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;">
            <div>
                <div style="font-family:'Space Mono',monospace; font-size:0.62rem; color:#00C8B4;
                            letter-spacing:0.1em; margin-bottom:0.5rem;">TARGET UTAMA</div>
                <div style="color:#E8EAF6; font-size:0.88rem; line-height:1.8;">
                    ▸ Segmen Champions (C3) — {models['cluster_profiles'].loc[3,'size']:,} nasabah<br>
                    ▸ Conv Rate historis: {models['cluster_profiles'].loc[3,'conversion_rate']:.1%}<br>
                    ▸ Channel: Cellular — efisiensi terbaik<br>
                    ▸ Bulan optimal: {df.groupby('month')['y'].mean().idxmax().upper()}
                </div>
            </div>
            <div>
                <div style="font-family:'Space Mono',monospace; font-size:0.62rem; color:#9B72F5;
                            letter-spacing:0.1em; margin-bottom:0.5rem;">PARAMETER EKSEKUSI</div>
                <div style="color:#E8EAF6; font-size:0.88rem; line-height:1.8;">
                    ▸ Threshold prediksi: ≥50% probabilitas<br>
                    ▸ Maks kontak: 3x per nasabah<br>
                    ▸ Durasi target: &gt;3 menit (180 dtk)<br>
                    ▸ Model AUC: {(models['test_results']['lr_auc']+models['test_results']['rf_auc'])/2:.4f}
                </div>
            </div>
        </div>
        <div style="margin-top:1.2rem; padding-top:1rem; border-top:1px solid rgba(108,59,219,0.2);">
            <div style="font-family:'Space Mono',monospace; font-size:0.6rem; color:#546E7A; line-height:1.6;">
                Referensi: Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success
                of bank telemarketing. Decision Support Systems, Elsevier, 62, 22–31. DOI: 10.1016/j.dss.2014.03.001
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="footer-bar"><p>Bank DSS Analytical Command Center | Kerangka Simon (1977): Intelligence → Design → Choice | DSS Project</p></div>', unsafe_allow_html=True)
