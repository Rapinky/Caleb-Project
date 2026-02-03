import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from PIL import Image

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

# Custom Styling (Professional Dashboard + Green Tone)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .main {
        background-color: #f8fafc;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Profile-style Branding Bar */
    .brand-container {
        display: flex;
        align-items: center;
        padding: 15px;
        background-color: #f1f5f9;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .brand-logo {
        width: 100px;
        height: 100px;
        object-fit: contain;
        margin-right: 12px;
    }
    .brand-text {
        display: flex;
        flex-direction: column;
    }
    .brand-group {
        font-size: 1.1em;
        font-weight: 700;
        color: #1e293b;
    }
    .brand-status {
        font-size: 0.8em;
        color: #22c55e; /* Logo Green */
        font-weight: 600;
    }

    /* Action Buttons (Logo Green) */
    .stButton>button {
        background-color: #22c55e !important; /* Green similar to logo */
        color: white !important;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        height: 45px;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #16a34a !important;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
    }
    
    /* Glass Cards for Main Content */
    .glass-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 700;
    }
    
    /* Tag Badge Styling */
    .word-box {
        background-color: white;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #f1f5f9;
    }

</style>
""", unsafe_allow_html=True)

# Sidebar Layout
with st.sidebar:
    # Profile-style Sidebar Header
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "caleb_logo.png")
    
    st.markdown(f"""
    <div class="brand-container">
        <img src="https://calebuniversity.edu.ng/wp-content/uploads/2021/05/logo-1.png" class="brand-logo" onerror="this.src='https://via.placeholder.com/80'">
        <div class="brand-text">
            <div class="brand-group">Group 13</div>
            <div class="brand-status">● Active Session</div>
        </div>
    </div>
    <div style="margin-top: -15px; padding-left: 5px; margin-bottom: 20px;">
        <div style="font-weight: 600; color: #1e3a8a; font-size: 0.9em;">Caleb University Imota</div>
        <div style="font-size: 0.8em; color: #64748b;">Computer Science Dept.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("⚙️ Configuration")
    
    data_source = st.selectbox(
        "Data Source", 
        ["NLTK Treebank", "Synthetic Fallback"], 
        index=0,
        help="Select the training dataset. 'NLTK' is high-accuracy. 'Synthetic' is for quick demos."
    )
    
    min_freq = st.slider(
        "UNK Threshold (min_freq)", 
        1, 10, 2,
        help="Words that appear fewer than this many times in the training data are treated as Unknown (<UNK>). This helps the model learn to predict tags for unseen words."
    )
    
    eval_size = st.slider(
        "Evaluation Size", 
        50, 1000, 200, step=50,
        help="The number of sentences from the test set to use for accuracy calculation. Larger sizes take longer to process."
    )
    
    st.markdown("---")
    st.subheader("📑 Options")
    show_mft = st.checkbox("MFT Baseline Comparison", value=True)
    show_cm_annot = st.checkbox("Detailed Matrix Annotations", value=False)
    
    st.markdown("---")
    st.subheader("🏷️ Tag Legend")
    st.markdown("""
    - **DET**: Determiner (the, a)
    - **NOUN**: Noun (cat, dog)
    - **VERB**: Verb (run, jump)
    - **ADJ**: Adjective (fast, blue)
    - **ADV**: Adverb (quickly)
    - **PRON**: Pronoun (it, she)
    - **ADP**: Adposition (in, on)
    """)

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

# App Header
st.title("🧩 Hidden Markov Model POS Tagger")
st.markdown("---")

# Initialize
use_synthetic = (data_source == "Synthetic Fallback")
with st.spinner("Preparing Models..."):
    hmm_model, mft_model, test_data, vocab = initialize_system(min_freq, use_synthetic)

# Main Multi-Tab View
tab1, tab2, tab3 = st.tabs(["🏷️ Live Tagger", "📈 Analytics", "⚖️ Theory"])

with tab1:
    st.markdown("### Interactive Tagging")
    input_text = st.text_area("Input Sentence", value="The quick brown fox jumps over the lazy dog", help="Type any sentence here to see how the HMM tags it.")
    
    if st.button("Tag Sentence"):
        if input_text:
            words = input_text.strip().split()
            with st.spinner("Analyzing sequence..."):
                hmm_preds = viterbi(words, hmm_model)
                
                # Results Table
                df = pd.DataFrame({"Word": words, "HMM Prediction": hmm_preds})
                if show_mft:
                    df["Baseline (MFT)"] = mft_predict(words, mft_model)
                
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Prediction Overview")
                st.dataframe(df, use_container_width=True)
                
                st.markdown("#### Visual Sequence")
                tag_colors = {
                    "NOUN": "#1e40af", "VERB": "#15803d", "ADJ": "#b45309", 
                    "DET": "#0369a1", "ADV": "#6d28d9", "PRON": "#be185d",
                    "ADP": "#374151", "CONJ": "#991b1b", ".": "#1e293b"
                }
                
                # Dynamic Columns for visualization
                cols = st.columns(len(words))
                for i, (w, t) in enumerate(zip(words, hmm_preds)):
                    with cols[i]:
                        color = tag_colors.get(t, "#475569")
                        st.markdown(f"""
                        <div class="word-box" style="border-top: 4px solid {color};">
                            <div style="font-weight: 700; color: #334155;">{w}</div>
                            <div style="color: {color}; font-size: 0.75em; font-weight: 800; text-transform: uppercase;">{t}</div>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some text.")

with tab2:
    st.markdown("### System Performance")
    if st.button("Start Global Evaluation"):
        with st.spinner(f"Testing on {eval_size} sentences..."):
            results = evaluate_models(test_data[:eval_size], hmm_model, mft_model)
            
            # Metrics Row
            m1, m2 = st.columns(2)
            with m1:
                st.metric("HMM Accuracy", f"{results['hmm_accuracy']:.2%}")
            with m2:
                if show_mft:
                    st.metric("Baseline Accuracy", f"{results['mft_accuracy']:.2%}", delta=f"{results['hmm_accuracy']-results['mft_accuracy']:.2%}")
            
            # Confusion Heatmap
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Confusion Matrix (HMM)")
            sns.set_style("white")
            fig, ax = plt.subplots(figsize=(10, 7))
            cm = results['confusion_matrix']
            tags = results['tags']
            norm_cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
            
            sns.heatmap(norm_cm, annot=show_cm_annot, cmap="Greens", xticklabels=tags, yticklabels=tags, ax=ax)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("### HMM Scientific Foundation")
    st.info("The HMM uses Transition and Emission matrices to find the most probable hidden states (tags) for observed symbols (words).")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transition Heatmap")
        fig_t, ax_t = plt.subplots(figsize=(8, 6))
        sns.heatmap(hmm_model["A"], xticklabels=hmm_model["idx2tag"].values(), yticklabels=hmm_model["idx2tag"].values(), cmap="Blues", ax=ax_t)
        st.pyplot(fig_t)
    with col2:
        st.subheader("Sample Probabilities")
        st.markdown("""
        **Viterbi Equation:**  
        `v[t][j] = max(v[t-1][i] * A[i][j]) * B[j][word]`
        
        This recursive approach ensures we find the globally optimal sequence instead of just the best tag for each word in isolation.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.8em; padding-bottom: 20px;">
    Caleb University Imota | Project Group 13 | Final Deliverable
</div>
""", unsafe_allow_html=True)
