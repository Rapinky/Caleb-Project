import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import base64

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from utils import download_nltk_data, load_pos_data, get_vocab_and_tags
from train_hmm import train_hmm
from evaluate import viterbi, train_mft_baseline, mft_predict, evaluate_models
from PIL import Image

# Favicon
icon = Image.open("assets/Caleb_logo.png")

st.set_page_config(
    page_title="Caleb Project",
    page_icon=icon,
    layout="wide"
)
# Page Configuration
st.set_page_config(
    page_title="Caleb University - HMM POS Tagger",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color Constants
LOGO_GREEN = "#22c55e"
LIGHT_GREEN = "#f0fdf4"
NAVY_BLUE = "#1e3a8a"
SLATE_600 = "#475569"

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Custom Styling
logo_base64 = ""
logo_path = os.path.join(os.path.dirname(__file__), "assets", "caleb_logo.png")
if os.path.exists(logo_path):
    logo_base64 = get_base64_of_bin_file(logo_path)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    
    /* Main Background */
    .stApp {{
        background-color: #f8fafc;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }}
    
    /* Profile-style Branding Bar */
    .brand-container {{
        display: flex;
        align-items: center;
        padding: 12px;
        background-color: {LIGHT_GREEN};
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #dcfce7;
    }}
    .brand-logo-img {{
        width: 50px;
        height: 50px;
        border-radius: 8px;
        object-fit: contain;
        background: white;
        padding: 4px;
        margin-right: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .brand-text-container {{
        display: flex;
        flex-direction: column;
    }}
    .brand-group-name {{
        font-size: 0.95em;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
    }}
    .brand-status-indicator {{
        font-size: 0.75em;
        color: {LOGO_GREEN};
        font-weight: 600;
        margin: 0;
    }}

    /* Sidebar Section Headers */
    .sidebar-section-header {{
        display: flex;
        align-items: center;
        margin-top: 25px;
        margin-bottom: 12px;
        padding-left: 5px;
    }}
    .sidebar-icon {{
        width: 20px;
        height: 20px;
        margin-right: 10px;
        color: {LOGO_GREEN};
    }}
    .sidebar-label {{
        font-size: 0.85em;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Action Buttons (Logo Green) */
    .stButton>button {{
        background-color: {LOGO_GREEN} !important;
        color: white !important;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        height: 42px;
        transition: all 0.2s ease;
        width: 100%;
        font-family: 'Inter', sans-serif;
    }}
    
    .stButton>button:hover {{
        background-color: #16a34a !important;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
    }}
    
    /* Segmented Control (Tabs as Toggles) */
    div[data-testid="stHorizontalBlock"] .stButton > button {{
        background-color: white !important;
        color: {SLATE_600} !important;
        border: 1px solid #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 0.9em !important;
    }}
    
    /* Target specifically the active tab via session state and CSS injection */
    .active-toggle button {{
        background-color: {LOGO_GREEN} !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(34, 197, 94, 0.2) !important;
    }}

    /* Content Cards */
    .content-card {{
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }}
    
    /* Word Highlight Boxes */
    .word-box {{
        background-color: white;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }}
</style>
""", unsafe_allow_html=True)

# SVG Icons
SVG_GEAR = f'<svg class="sidebar-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>'
SVG_LIST = f'<svg class="sidebar-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line></svg>'
SVG_TAG = f'<svg class="sidebar-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path><line x1="7" y1="7" x2="7.01" y2="7"></line></svg>'
SVG_TAB_TAG = '<svg style="width:16px; margin-right:8px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path></svg>'
SVG_TAB_CHART = '<svg style="width:16px; margin-right:8px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>'
SVG_TAB_BOOK = '<svg style="width:16px; margin-right:8px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>'

# Sidebar Header
with st.sidebar:
    logo_src = f"data:image/png;base64,{logo_base64}" if logo_base64 else "https://calebuniversity.edu.ng/wp-content/uploads/2021/05/logo-1.png"
    st.markdown(f"""
    <div class="brand-container">
        <img src="{logo_src}" class="brand-logo-img">
        <div class="brand-text-container">
            <p class="brand-group-name">Project Group 13</p>
            <p class="brand-status-indicator">Academic Edition</p>
        </div>
    </div>
    <div style="padding-left: 12px; margin-top: -10px; margin-bottom: 25px;">
        <p style="font-weight: 600; color: {NAVY_BLUE}; font-size: 0.85em; margin: 0;">Caleb University Imota</p>
        <p style="font-size: 0.75em; color: {SLATE_600}; margin: 0;">Computer Science Dept.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<div class="sidebar-section-header">{SVG_GEAR}<span class="sidebar-label">Configuration</span></div>', unsafe_allow_html=True)
    data_source = st.selectbox("Model Dataset", ["NLTK Treebank", "Synthetic Fallback"], index=0)
    min_freq = st.slider("UNK Sensitivity", 1, 10, 2, help="Lower = strict, Higher = flexible (handles more new words).")
    eval_size = st.slider("Evaluation Count", 50, 1000, 200, step=50, help="Number of sentences for accuracy calculation.")
    
    st.markdown(f'<div class="sidebar-section-header">{SVG_LIST}<span class="sidebar-label">Preferences</span></div>', unsafe_allow_html=True)
    show_mft = st.checkbox("Show Baseline", value=True)
    show_cm_annot = st.checkbox("Detailed Matrix", value=False)
    
    st.markdown(f'<div class="sidebar-section-header">{SVG_TAG}<span class="sidebar-label">Tag Legend</span></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size: 0.8em; background: {LIGHT_GREEN}; padding: 10px; border-radius: 8px; border: 1px solid #dcfce7; color: #166534;">
        <strong>DET</strong>: Determiner • <strong>NOUN</strong>: Noun<br>
        <strong>VERB</strong>: Verb • <strong>ADJ</strong>: Adjective<br>
        <strong>ADV</strong>: Adverb • <strong>PRON</strong>: Pronoun
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_system(min_freq, use_synthetic):
    if not use_synthetic:
        try: download_nltk_data()
        except: use_synthetic = True
    train_data, test_data = load_pos_data(use_synthetic=use_synthetic)
    vocab, tags = get_vocab_and_tags(train_data, min_freq=min_freq)
    hmm_model = train_hmm(train_data, vocab, tags)
    mft_model = train_mft_baseline(train_data, tags)
    return hmm_model, mft_model, test_data, vocab

# Main Content
st.title("Hidden Markov Model POS Tagger")
st.markdown(f"**Academic Research Tool** • *Statistical NLP Pipeline*")

# State Management for Toggle Tabs
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Tagger"

# Custom Segmented Toggle Row
toggle_col1, toggle_col2, toggle_col3, _ = st.columns([1, 1, 1, 3])

with toggle_col1:
    tagger_class = "active-toggle" if st.session_state.active_tab == "Tagger" else ""
    st.markdown(f'<div class="{tagger_class}">', unsafe_allow_html=True)
    if st.button(f"Tagger", use_container_width=True):
        st.session_state.active_tab = "Tagger"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with toggle_col2:
    analytics_class = "active-toggle" if st.session_state.active_tab == "Analytics" else ""
    st.markdown(f'<div class="{analytics_class}">', unsafe_allow_html=True)
    if st.button(f"Analytics", use_container_width=True):
        st.session_state.active_tab = "Analytics"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with toggle_col3:
    theory_class = "active-toggle" if st.session_state.active_tab == "Theory" else ""
    st.markdown(f'<div class="{theory_class}">', unsafe_allow_html=True)
    if st.button(f"Theory", use_container_width=True):
        st.session_state.active_tab = "Theory"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Initialize Engine
use_synthetic = (data_source == "Synthetic Fallback")
hmm_model, mft_model, test_data, vocab = initialize_system(min_freq, use_synthetic)

# Tab Content Rendering
if st.session_state.active_tab == "Tagger":
    st.markdown("### Interactive Tagging")
    input_text = st.text_area("Input Sentence", value="The quick brown fox jumps over the lazy dog")
    
    if st.button("Tag Sentence"):
        if input_text:
            words = input_text.strip().split()
            hmm_preds = viterbi(words, hmm_model)
            
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.subheader("Results")
            df = pd.DataFrame({"Word": words, "Prediction": hmm_preds})
            if show_mft: df["Baseline"] = mft_predict(words, mft_model)
            st.dataframe(df, use_container_width=True)
            
            st.markdown("#### Annotated View")
            visual_cols = st.columns(min(len(words), 8))
            for i, (w, t) in enumerate(zip(words, hmm_preds)):
                color = {"NOUN": "#1e40af", "VERB": "#15803d"}.get(t, SLATE_600)
                with visual_cols[i % 8]:
                    st.markdown(f'<div class="word-box" style="border-top: 3px solid {color};"><div style="font-weight:700; color:{NAVY_BLUE};">{w}</div><div style="color:{color}; font-size:0.75em; font-weight:800;">{t}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.active_tab == "Analytics":
    st.markdown("### Model Analytics")
    if st.button("Run Evaluation"):
        results = evaluate_models(test_data[:eval_size], hmm_model, mft_model)
        m1, m2 = st.columns(2)
        m1.metric("HMM Accuracy", f"{results['hmm_accuracy']:.2%}")
        if show_mft: m2.metric("Baseline Accuracy", f"{results['mft_accuracy']:.2%}")
        
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(results['confusion_matrix'], annot=show_cm_annot, cmap="Greens", ax=ax)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.active_tab == "Theory":
    st.markdown("### Probabilistic Logic")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transition Matrix")
        fig_t, ax_t = plt.subplots(figsize=(8, 6))
        sns.heatmap(hmm_model["A"], cmap="Blues", ax=ax_t)
        st.pyplot(fig_t)
    with col2:
        st.subheader("Emission Subset")
        fig_e, ax_e = plt.subplots(figsize=(8, 6))
        sns.heatmap(hmm_model["B"][:, :15], cmap="Purples", ax=ax_e)
        st.pyplot(fig_e)

# Footer
st.markdown(f'<div style="text-align: center; color: #94a3b8; font-size: 0.75em; border-top: 1px solid #e2e8f0; padding-top: 20px;">Caleb University • Group 13 Deliverable</div>', unsafe_allow_html=True)
