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

# Custom Styling (Academic Premium + Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Glassmorphism Card style */
    .stApp > header {
        background-color: transparent;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        margin-bottom: 20px;
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(10px);
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.6);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    h1, h2, h3 {
        color: #1e3a8a; /* Deep Navy */
        font-weight: 700;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #1e3a8a, #3b82f6);
        color: white;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
    }

    /* Tag Badge Styling */
    .tag-badge {
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: bold;
        color: white;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Branding
with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "caleb_logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="font-size: 1.2em; color: #1e3a8a; margin-bottom: 5px;">Caleb University</h2>
        <p style="font-size: 0.9em; color: #666;">Imota, Lagos State</p>
        <p style="font-size: 0.8em; font-weight: bold; color: #3b82f6;">Dept. of Computer Science</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("🛠️ Configuration")
    data_source = st.selectbox("Data Source", ["NLTK Treebank", "Synthetic Fallback"], index=0)
    min_freq = st.slider("UNK Threshold (min_freq)", 1, 5, 2)
    eval_size = st.slider("Evaluation Subset Size", 50, 500, 200, step=50)
    show_mft = st.checkbox("Compare with MFT Baseline", value=True)
    show_cm_annot = st.checkbox("Annotate Confusion Matrix", value=False)
    
    st.markdown("---")
    st.subheader("🏷️ Tag Legend")
    st.markdown("""
    - **DET**: Determiner (the, a, that)
    - **NOUN**: Noun (cat, dog, lecture)
    - **VERB**: Verb (runs, kicked, is)
    - **ADJ**: Adjective (quick, lazy)
    - **ADV**: Adverb (quickly, softly)
    - **PRON**: Pronoun (it, they)
    - **ADP**: Adposition (on, in, with)
    - **CONJ**: Conjunction (and, but)
    - **PRT**: Particle (up, out)
    - **.** : Punctuation
    """)
    
    st.markdown("---")
    st.info("""
    **Note on Baseline (MFT):**
    The MFT baseline simply picks the single most common tag seen in training. It serves as a benchmark for HMM.
    """)

@st.cache_resource
def initialize_system(min_freq, use_synthetic):
    """Load data and train models only once."""
    if not use_synthetic:
        try:
            download_nltk_data()
        except:
            st.warning("Could not reach NLTK servers. Falling back to synthetic mode.")
            use_synthetic = True
            
    train_data, test_data = load_pos_data(use_synthetic=use_synthetic)
    vocab, tags = get_vocab_and_tags(train_data, min_freq=min_freq)
    
    hmm_model = train_hmm(train_data, vocab, tags)
    mft_model = train_mft_baseline(train_data, tags)
    
    return hmm_model, mft_model, test_data, vocab

# App Header
st.title("🧠 Hidden Markov Model POS Tagger")
st.markdown("---")

# Initialize
use_synthetic = (data_source == "Synthetic Fallback")
with st.spinner("Initializing models (Training HMM)..."):
    hmm_model, mft_model, test_data, vocab = initialize_system(min_freq, use_synthetic)

# Tabs
tab1, tab2, tab3 = st.tabs(["🏷️ Live Tagger", "📊 Performance Analysis", "🎓 Academic Deep Dive"])

with tab1:
    st.header("Interactive Tagging")
    
    if use_synthetic:
        st.warning("""
        ⚠️ **Running in Synthetic Fallback Mode**
        The model is using a smaller sample set. Transition repetition may occur for unknown words.
        """)
        
    input_text = st.text_area("Enter a sentence to analyze:", placeholder="The old man the boats", value="The quick brown fox jumps over the lazy dog", height=100)
    
    if st.button("Analyze Sentence"):
        if input_text:
            words = input_text.strip().split()
            with st.spinner("Viterbi decoding..."):
                hmm_preds = viterbi(words, hmm_model)
                
                results_data = {"Word": words, "Predicted POS (HMM)": hmm_preds}
                if show_mft:
                    mft_preds = mft_predict(words, mft_model)
                    results_data["Baseline (MFT)"] = mft_preds
                
                df = pd.DataFrame(results_data)
                
                with st.container():
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("POS Prediction Result")
                    st.dataframe(df, use_container_width=True)
                    
                    st.markdown("### Tagged Sequence")
                    tag_colors = {
                        "NOUN": "#1e3a8a", "VERB": "#10b981", "ADJ": "#f59e0b", 
                        "DET": "#3b82f6", "ADV": "#8b5cf6", "PRON": "#ec4899",
                        "ADP": "#6b7280", "CONJ": "#f43f5e", "PRT": "#06b6d4",
                        ".": "#4b5563"
                    }
                    
                    # Horizontal Layout
                    cols = st.columns(len(words))
                    for i, (w, t) in enumerate(zip(words, hmm_preds)):
                        with cols[i]:
                            color = tag_colors.get(t, "#4b5563")
                            st.markdown(f"""
                            <div style="background-color: white; padding: 10px; border-radius: 10px; text-align: center; border-top: 5px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                                <div style="font-weight: bold; color: #333;">{w}</div>
                                <div style="font-size: 0.8em; color: {color}; font-weight: 700;">{t}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                # HMM Insights
                with st.expander("🔍 HMM Decoding Insights"):
                    oov_words = [w for w in words if w.lower() not in hmm_model['word2idx']]
                    st.write(f"**Out-of-Vocabulary (OOV) Words:** {', '.join(oov_words) if oov_words else 'None'}")
                    st.markdown("""
                    **The Viterbi Strategy:**  
                    For each word, the model calculates:  
                    `P(State | Prev State) * P(Word | State)`  
                    Even if the word is unknown, the **Transition Probabilities** (learned from grammar patterns) guide the model to the most logical sequence.
                    """)
        else:
            st.warning("Please enter a sentence.")

with tab2:
    st.header("Evaluation Metrics")
    
    if st.button("Run Evaluation"):
        with st.spinner(f"Evaluating on {eval_size} sentences..."):
            results = evaluate_models(test_data[:eval_size], hmm_model, mft_model)
            
            # Metrics
            c1, c2 = st.columns(2)
            with c1:
                st.metric("HMM Accuracy", f"{results['hmm_accuracy']:.2%}", help="Higher is better. Reflects context-aware tagging.")
            with c2:
                if show_mft:
                    st.metric("Baseline Accuracy", f"{results['mft_accuracy']:.2%}", delta=f"{results['hmm_accuracy']-results['mft_accuracy']:.2%}", help="The baseline doesn't use sequence context.")
            
            # Confusion Matrix
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Confusion Matrix")
            cm = results['confusion_matrix']
            tags = results['tags']
            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm_norm, annot=show_cm_annot, fmt=".2f", cmap="Blues", 
                        xticklabels=tags, yticklabels=tags, ax=ax)
            ax.set_title("Normalized HMM Confusion Matrix")
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Error List
            st.subheader("Top Model Confusions")
            errors = []
            for i in range(len(tags)):
                for j in range(len(tags)):
                    if i != j and cm[i, j] > 0:
                        errors.append({"True Tag": tags[i], "Predicted Tag": tags[j], "Count": cm[i, j]})
            
            error_df = pd.DataFrame(errors).sort_values("Count", ascending=False).head(10)
            st.table(error_df)

with tab3:
    st.header("🎓 HMM Theoretical Foundation")
    st.markdown("""
    <div class="glass-card">
    <h3>The Hidden Markov Model</h3>
    <p>A Hidden Markov Model (HMM) is a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobserved (hidden) states.</p>
    <ul>
        <li><strong>States:</strong> The Part-of-Speech tags (NOUN, VERB, etc.)</li>
        <li><strong>Observations:</strong> The words in the sentence.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Transition Matrix (A)")
        st.info("Likelihood of one tag following another.")
        fig_a, ax_a = plt.subplots(figsize=(8, 6))
        sns.heatmap(hmm_model["A"], xticklabels=hmm_model["idx2tag"].values(), yticklabels=hmm_model["idx2tag"].values(), cmap="YlGnBu", ax=ax_a)
        st.pyplot(fig_a)

    with col_b:
        st.subheader("Emission Matrix (B - Subset)")
        st.info("Likelihood of a tag producing a specific word.")
        # Show top words for some tags
        subset_words = sorted(list(vocab))[:20]
        fig_b, ax_b = plt.subplots(figsize=(8, 6))
        sns.heatmap(hmm_model["B"][:, :20], xticklabels=subset_words, yticklabels=hmm_model["idx2tag"].values(), cmap="Purples", ax=ax_b)
        st.pyplot(fig_b)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 0.8em; padding: 20px;">
    © 2026 Caleb University - Computer Science Department | Project Group 13<br>
    Built with ❤️ and Python (Streamlit + NLTK)
</div>
""", unsafe_allow_html=True)
