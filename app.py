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
    page_title="HMM POS Tagger",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🛠️ Configuration")
data_source = st.sidebar.selectbox("Data Source", ["NLTK Treebank", "Synthetic Fallback"], index=0)
min_freq = st.sidebar.slider("UNK Threshold (min_freq)", 1, 5, 2)
eval_size = st.sidebar.slider("Evaluation Subset Size", 50, 500, 200, step=50)
show_mft = st.sidebar.checkbox("Compare with MFT Baseline", value=True)
show_cm_annot = st.sidebar.checkbox("Annotate Confusion Matrix", value=False)

# Tag Legend in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🏷️ Tag Legend")
st.sidebar.markdown("""
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

# Explanation Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.info("""
**Note on Baseline (MFT):**
The MFT baseline simply picks the single most common tag seen in training. In 'Synthetic Fallback' mode, the training data is extremely small, so the baseline often defaults to 'DET' for unknown words.
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
tab1, tab2 = st.tabs(["🏷️ Live Tagger", "📊 Performance Analysis"])

# Tab 1: Live Tagger
with tab1:
    st.header("Interactive Tagging")
    
    if use_synthetic:
        st.warning("""
        ⚠️ **Running in Synthetic Fallback Mode**
        Because the real NLTK data couldn't be loaded, the model is trained on only **11 sample sentences**.
        - **Known words**: the, dog, runs, cat, man, woman, i, she, they, it, we, book, friend, key, table, see, wow, and, fox, is, am, are, was, were, put, on, in, happy, small, very, fast, quick, brown, slowly, quickly, jumps, over, lazy, who, that, this, test.
        - **Unknown words**: Any word not listed above will rely purely on 'grammar patterns' (transition probabilities). 
        - **Repetition**: If you enter many unknown words, the model will repeat the most likely tag sequence it knows (e.g., DET -> NOUN -> VERB).
        """)
        st.info("**Try these 'Safe' sentences:** 'The dog runs slowly', 'I see you', 'They walk on the grass'")

    input_text = st.text_area("Enter a sentence to analyze:", placeholder="The old man the boats", value="The old man the boats", height=100)
    
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
                
                st.subheader("Results Table")
                st.dataframe(df, use_container_width=True)
                
                # HMM Insights
                with st.expander("🔍 HMM Insights (How it was calculated)"):
                    oov_words = [w for w in words if w.lower() not in hmm_model['word2idx']]
                    st.write(f"**Unknown Words (OOV):** {', '.join(oov_words) if oov_words else 'None'}")
                    st.info("""
                    Even for unknown words (like 'Lecturer'), the HMM uses **Transition Probabilities** 
                    learned during training (e.g., *'Determiners are usually followed by Nouns'*) 
                    to guess the most likely tag sequence.
                    """)

                # Visual highlight version
                st.markdown("### Tagged Sentence")
                tag_colors = {
                    "NOUN": "#e1f5fe", "VERB": "#fff3e0", "ADJ": "#f3e5f5", 
                    "DET": "#e8f5e9", "ADV": "#fce4ec", "PRON": "#e0f2f1",
                    "ADP": "#ede7f6", "CONJ": "#fffde7", "PRT": "#efebe9",
                    ".": "#f5f5f5"
                }
                
                cols = st.columns(len(words))
                for i, (w, t) in enumerate(zip(words, hmm_preds)):
                    with cols[i]:
                        color = tag_colors.get(t, "#ffffff")
                        st.markdown(f"""
                        <div style="background-color:{color}; padding:10px; border-radius:5px; text-align:center; border: 1px solid #ddd;">
                            <div style="font-weight:bold; color:#333;">{w}</div>
                            <div style="font-size:0.8em; color:#666;">{t}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a sentence.")

with tab2:
    st.header("Evaluation Metrics")
    
    if st.button("Run Evaluation"):
        with st.spinner(f"Evaluating on {eval_size} test sentences..."):
            results = evaluate_models(test_data[:eval_size], hmm_model, mft_model)
            
            # Metrics
            c1, c2 = st.columns(2)
            with c1:
                st.metric("HMM Accuracy", f"{results['hmm_accuracy']:.2%}")
            with c2:
                if show_mft:
                    st.metric("MFT Baseline Accuracy", f"{results['mft_accuracy']:.2%}")
            
            # Confusion Matrix
            st.markdown("---")
            st.subheader("Confusion Matrix")
            cm = results['confusion_matrix']
            tags = results['tags']
            
            # Normalize
            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm_norm, annot=show_cm_annot, fmt=".2f", cmap="Greens", 
                        xticklabels=tags, yticklabels=tags, ax=ax)
            ax.set_title("Normalized HMM Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)
            
            # Error List
            st.subheader("Top Tag Confusions")
            errors = []
            for i in range(len(tags)):
                for j in range(len(tags)):
                    if i != j and cm[i, j] > 0:
                        errors.append({"True Tag": tags[i], "Pred Tag": tags[j], "Count": cm[i, j]})
            
            error_df = pd.DataFrame(errors).sort_values("Count", ascending=False).head(10)
            st.table(error_df)

# Footer
st.markdown("---")
st.caption(f"Group 13 - Hidden Markov Model POS Tagging Project | Data Source: {data_source}")
