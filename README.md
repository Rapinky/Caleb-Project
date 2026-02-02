# HMM POS Tagger — Group 13

A robust Hidden Markov Model (HMM) implementation for Part-of-Speech (POS) tagging, featuring numerical stability (log-space Viterbi), unknown word handling, and a professional web interface.

## Key Features
- **Custom HMM Implementation**: Hand-coded transition, emission, and initial probability estimation.
- **Log-Space Viterbi**: Prevents numerical underflow in long sentences.
- **Robust UNK Strategy**: Probability-based handling for unknown words using frequency filtering.
- **Synthetic Fallback**: Automatically switches to a mock corpus if NLTK data is unavailable.
- **Professional Web UI**: Interactive tagging and performance dashboards built with Streamlit.

## Setup & Installation

1. **Install Python 3.10+** (Recommended: 3.11-3.13; tested on 3.14).
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Data** (Optional/Auto-handled):
   The system will attempt to download NLTK data automatically. If connectivity fails, you can use the built-in **Synthetic Fallback** mode or run the manual setup:
   ```bash
   python src/setup_data.py
   ```

## Usage

### 🌐 Web Interface (Recommended)
Experience the tagger with a full GUI, color-coded results, and live performance metrics. Includes a toggle to switch between the full Treebank and Synthetic modes:
```bash
python -m streamlit run app.py
```

### 🚀 Interactive CLI
Run the interactive terminal loop:
```bash
python src/interactive_tagger.py
```

### 📊 Evaluation & Analysis
See the full research notebook for detailed math, training logs, and advanced charts:
```bash
jupyter notebook notebooks/test_hmm.ipynb
```

---

## Technical Details
For a deep dive into the implementation math, design decisions, and linguistic performance analysis, see the [Walkthrough](./walkthrough.md).
