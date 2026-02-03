import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from utils import download_nltk_data, load_pos_data, get_vocab_and_tags
from train_hmm import train_hmm
from evaluate import viterbi, train_mft_baseline, mft_predict, evaluate_models

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

# Custom Styling (Professional Dashboard + SVG Icons)
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
        font-size: 0.9em;
        font-weight: 600;
        color: #334155;
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
    }}
    
    .stButton>button:hover {{
        background-color: #16a34a !important;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
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
        transition: transform 0.2s;
    }}
    .word-box:hover {{
        transform: translateY(-2px);
        border-color: {LOGO_GREEN};
    }}

    /* Remove default Streamlit padding at top */
    .block-container {{
        padding-top: 2rem !important;
    }}
</style>
""", unsafe_allow_html=True)

# SVG Icons Definitions
SVG_GEAR = f'<svg class="sidebar-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>'
SVG_LIST = f'<svg class="sidebar-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line></svg>'
SVG_TAG = f'<svg class="sidebar-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path><line x1="7" y1="7" x2="7.01" y2="7"></line></svg>'
SVG_CHART = f'<svg class="sidebar-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>'
SVG_BOOK = f'<svg class="sidebar-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>'

# Sidebar Layout
with st.sidebar:
    # 1. Profile-style Branding Bar (Logo Left, Text Right)
    st.markdown(f"""
    <div class="brand-container">
        <img src="https://calebuniversity.edu.ng/wp-content/uploads/2021/05/logo-1.png" class="brand-logo-img">
        <div class="brand-text-container">
            <p class="brand-group-name">Project Group 13</p>
            <p class="brand-status-indicator">Academic Edition</p>
        </div>
    </div>
    <div style="padding-left: 12px; margin-top: -10px; margin-bottom: 20px;">
        <p style="font-weight: 600; color: {NAVY_BLUE}; font-size: 0.85em; margin: 0;">Caleb University Imota</p>
        <p style="font-size: 0.75em; color: {SLATE_600}; margin: 0;">Computer Science Dept.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Configuration Section with SVG Gear
    st.markdown(f'<div class="sidebar-section-header">{SVG_GEAR}<span class="sidebar-label">Configuration</span></div>', unsafe_allow_html=True)
    
    data_source = st.selectbox(
        "Model Dataset", 
        ["NLTK Treebank", "Synthetic Fallback"], 
        index=0,
        help="NLTK: Realistic data (slow). Synthetic: Small sample (fast)."
    )
    
    min_freq = st.slider(
        "UNK Threshold", 
        1, 10, 2,
        help="Sets 'Unknown' word sensitivity. Helps model generalize to new text."
    )
    
    eval_size = st.slider(
        "Evaluation Size", 
        50, 1000, 200, step=50,
        help="Number of test samples for performance calculation."
    )
    
    # 3. Options Section with SVG List
    st.markdown(f'<div class="sidebar-section-header">{SVG_LIST}<span class="sidebar-label">Preferences</span></div>', unsafe_allow_html=True)
    show_mft = st.checkbox("Show MFT Baseline", value=True)
    show_cm_annot = st.checkbox("Matrix Annotations", value=False)
    
    # 4. Tag Legend Section with SVG Tag
    st.markdown(f'<div class="sidebar-section-header">{SVG_TAG}<span class="sidebar-label">Tag Legend</span></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size: 0.85em; background: {LIGHT_GREEN}; padding: 10px; border-radius: 8px; border: 1px solid #dcfce7;">
        <span style="color: {NAVY_BLUE}; font-weight: 600;">DET</span>: Determiner (the, a)<br>
        <span style="color: {NAVY_BLUE}; font-weight: 600;">NOUN</span>: Noun (cat, dog)<br>
        <span style="color: {NAVY_BLUE}; font-weight: 600;">VERB</span>: Verb (run, jump)<br>
        <span style="color: {NAVY_BLUE}; font-weight: 600;">ADJ</span>: Adjective (fast, blue)<br>
        <span style="color: {NAVY_BLUE}; font-weight: 600;">ADV</span>: Adverb (quickly)<br>
        <span style="color: {NAVY_BLUE}; font-weight: 600;">PRON</span>: Pronoun (it, she)
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_system(min_freq, use_synthetic):
    """Load data and train models only once."""
    if not use_synthetic:
        try:
            download_nltk_data()
        except:
            st.warning("Network issue: Falling back to Synthetic Mode.")
            use_synthetic = True
            
    train_data, test_data = load_pos_data(use_synthetic=use_synthetic)
    vocab, tags = get_vocab_and_tags(train_data, min_freq=min_freq)
    
    hmm_model = train_hmm(train_data, vocab, tags)
    mft_model = train_mft_baseline(train_data, tags)
    
    return hmm_model, mft_model, test_data, vocab

# Main Content Header
col_title, col_logo = st.columns([8, 2])
with col_title:
    st.title("Hidden Markov Model POS Tagger")
    st.markdown(f"**Academic Research Tool** • *Statistical NLP Pipeline*")

# Initialize
use_synthetic = (data_source == "Synthetic Fallback")
with st.spinner("Preparing Processing Engine..."):
    hmm_model, mft_model, test_data, vocab = initialize_system(min_freq, use_synthetic)

# Custom Tab Display (Professional Icons in Titles)
tab1, tab2, tab3 = st.tabs(["🏷️ Live Tagger", "📊 Analytics", "📖 Theory"])

with tab1:
    st.markdown("### Interactive Sequence Processing")
    input_text = st.text_area("Input Text", value="The quick brown fox jumps over the lazy dog", help="Analyze any sentence using the Viterbi algorithm.")
    
    if st.button("Calculate Optimal Tags"):
        if input_text:
            words = input_text.strip().split()
            with st.spinner("Running Viterbi decoding..."):
                hmm_preds = viterbi(words, hmm_model)
                
                # Results Display
                df = pd.DataFrame({"Word": words, "HMM Prediction": hmm_preds})
                if show_mft:
                    df["Baseline (MFT)"] = mft_predict(words, mft_model)
                
                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                st.subheader("Statistical Prediction Results")
                st.dataframe(df, use_container_width=True)
                
                # Visual highlight (New Design)
                st.markdown("#### Annotated Sequence")
                tag_colors = {{
                    "NOUN": "#1e40af", "VERB": "#15803d", "ADJ": "#b45309", 
                    "DET": "#0369a1", "ADV": "#6d28d9", "PRON": "#be185d",
                    "ADP": "#374151", "CONJ": "#991b1b", ".": "#1e293b"
                }}
                
                cols = st.columns(min(len(words), 8)) # Cap columns per row
                for i, (w, t) in enumerate(zip(words, hmm_preds)):
                    with cols[i % 8]:
                        color = tag_colors.get(t, "#475569")
                        st.markdown(f"""
                        <div class="word-box" style="border-bottom: 3px solid {color};">
                            <div style="font-weight: 700; color: #1e293b; font-size: 0.9em;">{w}</div>
                            <div style="color: {color}; font-size: 0.7em; font-weight: 800; margin-top: 4px;">{t}</div>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                with st.expander("🔍 Processing Insights"):
                    st.info(f"Analyzed {len(words)} tokens. System used Log-Space Viterbi to find the most probable hidden state path.")
        else:
            st.warning("Please enter some text for analysis.")

with tab2:
    st.markdown("### Model Performance Analytics")
    if st.button("Execute Validation Suite"):
        with st.spinner(f"Processing {eval_size} validation samples..."):
            results = evaluate_models(test_data[:eval_size], hmm_model, mft_model)
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("HMM Global Accuracy", f"{results['hmm_accuracy']:.2%}")
            with c2:
                if show_mft:
                    st.metric("Baseline Accuracy", f"{results['mft_accuracy']:.2%}", delta=f"{results['hmm_accuracy']-results['mft_accuracy']:.2%}")
            
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.subheader("HMM Confusion Matrix")
            fig, ax = plt.subplots(figsize=(10, 6))
            cm = results['confusion_matrix']
            tags = results['tags']
            norm_cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
            sns.heatmap(norm_cm, annot=show_cm_annot, cmap="Greens", xticklabels=tags, yticklabels=tags, ax=ax)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("### Probabilistic Foundations")
    st.info("The HMM is defined by states (POS tags) and observations (words). These heatmaps show the learned probabilities.")
    
    col_t, col_e = st.columns(2)
    with col_t:
        st.subheader("Transition Matrix")
        fig_t, ax_t = plt.subplots(figsize=(8, 6))
        sns.heatmap(hmm_model["A"], xticklabels=hmm_model["idx2tag"].values(), yticklabels=hmm_model["idx2tag"].values(), cmap="Blues", ax=ax_t)
        st.pyplot(fig_t)
    with col_e:
        st.subheader("Emission Subset")
        fig_e, ax_e = plt.subplots(figsize=(8, 6))
        sns.heatmap(hmm_model["B"][:, :15], xticklabels=sorted(list(vocab))[:15], yticklabels=hmm_model["idx2tag"].values(), cmap="Purples", ax=ax_e)
        st.pyplot(fig_e)

# Standard Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #94a3b8; font-size: 0.75em; padding-bottom: 30px;">
    <strong>Caleb University Imota</strong> • Computer Science Research Project • Group 13 Deliverable
</div>
""", unsafe_allow_html=True)
