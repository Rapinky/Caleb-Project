# HMM POS Tagger — Detailed Code Explanation

This document provides a comprehensive breakdown of the project codebase. It is designed to help you and your team understand, explain, and present the system architecture and implementation details.

---

## 🏗️ System Architecture Overview
The system is built as a modular pipeline:
1.  **Data Layer** (`utils.py`): Handles data loading, vocabulary extraction, and synthetic fallbacks.
2.  **Training Layer** (`train_hmm.py`): Estimates HMM parameters (A, B, Pi) using MLE with Laplacian smoothing.
3.  **Inference & Eval Layer** (`evaluate.py`): Implements the Log-Space Viterbi algorithm and performance metrics.
4.  **Interface Layer** (`app.py`, `interactive_tagger.py`): Provides Web and CLI interfaces for users.

---

## 📂 File-by-File Breakdown

### 1. `src/utils.py` — The Foundation
This file handles all data-related tasks.

-   **`download_nltk_data()`**: Attempts to download the `treebank` corpus. It uses `raise_on_error=False` to ensure the app doesn't crash if there's no internet.
-   **`generate_synthetic_data()`**: This is our "Safety Net." If the real data is missing, it provides a hardcoded list of 11 diverse sentences. We added many common words (pronouns, verbs, etc.) here to make the "offline" demo smarter.
-   **`load_pos_data()`**: The gatekeeper. It tries to load the real Treebank but automatically switches to `generate_synthetic_data()` if it fails. It also splits the data into **Training** (80%) and **Test** (20%) sets.
-   **`get_vocab_and_tags()`**: Scans the training data to build two lists: every unique word (Vocabulary) and every unique tag (Tags).
    -   *Crucial Detail*: It maps rare words (frequency < `min_freq`) to an `<UNK>` token to help the model handle unknown words later.

---

### 2. `src/train_hmm.py` — The Brain Builder
This file implements the math to "train" the Hidden Markov Model.

-   **`train_hmm()`**: This is where the model learns from the training data.
    -   **`Pi` (Initial Probabilities)**: Calculates how likely a sentence is to start with a specific tag (e.g., "How often do sentences start with a Noun?").
    -   **`A` (Transition Matrix)**: Calculates the probability of one tag following another (e.g., "If I just saw a Determiner, how likely is the next word to be a Noun?").
    -   **`B` (Emission Matrix)**: Calculates the probability of a word being associated with a tag (e.g., "If the tag is VERB, how likely is the word to be 'runs'?").
    -   **Laplacian Smoothing**: Every count starts at `1` (not `0`) to ensure we never have a "zero probability" which would break our math later.

---

### 3. `src/evaluate.py` — The Inference Engine
This is the most mathematically complex part of the project.

-   **`viterbi()`**: Implements the **Dynamic Programming** algorithm to find the most likely sequence of tags.
    -   **Log-Space Math**: We use `np.log()` instead of raw decimals. Why? Because multiplying small probabilities (e.g., 0.001 * 0.002) results in numbers so tiny that computers "forget" them (underflow). Adding logs (`log(a) + log(b)`) stays stable.
    -   **Backpointer**: As it moves forward through a sentence, it keeps track of "where it came from" so it can backtrack at the end to find the best path.
-   **`train_mft_baseline()`**: A "naive" model that just tags a word with its most frequent tag from training. We use this to prove that our HMM is actually smarter!
-   **`evaluate_models()`**: Compares the HMM vs the Baseline on the **Test Set** and calculates the final Accuracy score.

---

### 4. `app.py` — The Web Interface
Built with Streamlit, this turns the scripts into a professional application.

-   **`@st.cache_resource`**: This is a performance booster. It ensures the model is trained only **once** and kept in memory, so the UI responds instantly when you click "Analyze."
-   **Tabs**: Separates the **Live Tagger** (inference) from the **Performance Analysis** (testing).
-   **Color Highlighting**: Maps specific tags (NOUN, VERB) to colors to make the output human-readable and visually impressive.
-   **Confusion Matrix**: Uses `seaborn` and `matplotlib` to visualize exactly where the model gets confused (e.g., "Does it mistake Adjectives for Nouns?").

---

### 5. `src/setup_data.py` — The Troubleshooting Tool
A utility script we created to bypass NLTK connection errors.

-   **Manual Bypass**: It provides the team with direct links and instructions to download the corpus via a browser if Python's `requests` library is blocked by a firewall.

---

## 💡 Key Talking Points for your Team
1.  **Robustness**: "We implemented log-space Viterbi to prevent numerical underflow, which is a common failure point in simpler HMMs."
2.  **Graceful Degradation**: "The system detects network issues and switches to a synthetic corpus so that our demo never crashes."
3.  **Ambiguity Selection**: "Our HMM doesn't just look at the word; it looks at the *sequence*. That's why it can distinguish 'book' (NOUN) from 'book' (VERB) based on the surrounding tags."
