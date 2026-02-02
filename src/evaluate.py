"""
Module for evaluating HMM POS Tagger using log-space Viterbi and baseline comparisons.
"""

import numpy as np
from collections import Counter, defaultdict

def viterbi(sentence, model):
    """
    Log-space Viterbi algorithm to find the most likely tag sequence.
    Prevents underflow by working in log-probability space.
    """
    # Convert to logs, adding a small epsilon to avoid log(0)
    eps = 1e-10
    logA = np.log(model["A"] + eps)
    logB = np.log(model["B"] + eps)
    logPi = np.log(model["Pi"] + eps)
    
    tag2idx = model["tag2idx"]
    word2idx = model["word2idx"]
    idx2tag = model["idx2tag"]
    
    num_tags = len(tag2idx)
    num_words = len(sentence)
    
    # dp[i, j] stores the max log probability
    dp = np.full((num_tags, num_words), -np.inf)
    backpointer = np.zeros((num_tags, num_words), dtype=int)
    
    # Initialization
    first_word = sentence[0].lower()
    first_w_idx = word2idx.get(first_word, word2idx['<UNK>'])
    
    for t in range(num_tags):
        dp[t, 0] = logPi[t] + logB[t, first_w_idx]
        
    # Recursion
    for j in range(1, num_words):
        word = sentence[j].lower()
        w_idx = word2idx.get(word, word2idx['<UNK>'])
        
        for t in range(num_tags):
            # log(dp_prev * A * B) = log(dp_prev) + log(A) + log(B)
            # Find best tag from previous step
            probs = dp[:, j-1] + logA[:, t] + logB[t, w_idx]
            dp[t, j] = np.max(probs)
            backpointer[t, j] = np.argmax(probs)
            
    # Backtracking
    best_last_tag = np.argmax(dp[:, -1])
    path = [best_last_tag]
    for j in range(num_words - 1, 0, -1):
        path.append(backpointer[path[-1], j])
        
    path.reverse()
    return [idx2tag[t_idx] for t_idx in path]

def train_mft_baseline(train_data, tags):
    """
    Train a Most-Frequent-Tag baseline.
    """
    word_tag_counts = defaultdict(Counter)
    global_tag_counts = Counter()
    
    for sentence in train_data:
        for word, tag in sentence:
            word_tag_counts[word.lower()][tag] += 1
            global_tag_counts[tag] += 1
            
    mft_vocab = {word: counts.most_common(1)[0][0] for word, counts in word_tag_counts.items()}
    global_mft = global_tag_counts.most_common(1)[0][0]
    
    return mft_vocab, global_mft

def mft_predict(sentence, mft_model):
    mft_vocab, global_mft = mft_model
    return [mft_vocab.get(word.lower(), global_mft) for word in sentence]

def evaluate_models(test_data, hmm_model, mft_model):
    """
    Evaluate HMM and MFT models, return accuracies and confusion matrix data.
    """
    hmm_correct = 0
    mft_correct = 0
    total = 0
    
    tag2idx = hmm_model["tag2idx"]
    idx2tag = hmm_model["idx2tag"]
    num_tags = len(tag2idx)
    conf_matrix = np.zeros((num_tags, num_tags), dtype=int)
    
    # Ensure tags are in index order
    ordered_tags = [idx2tag[i] for i in range(num_tags)]
    
    for sentence in test_data:
        words = [w for w, t in sentence]
        true_tags = [t for w, t in sentence]
        
        hmm_preds = viterbi(words, hmm_model)
        mft_preds = mft_predict(words, mft_model)
        
        for h, m, t in zip(hmm_preds, mft_preds, true_tags):
            if h == t: hmm_correct += 1
            if m == t: mft_correct += 1
            
            # Populate confusion matrix for HMM: [True, Predicted]
            conf_matrix[tag2idx[t], tag2idx[h]] += 1
            total += 1
            
    return {
        "hmm_accuracy": hmm_correct / total,
        "mft_accuracy": mft_correct / total,
        "confusion_matrix": conf_matrix,
        "tags": ordered_tags
    }

if __name__ == "__main__":
    from utils import load_pos_data, get_vocab_and_tags
    from train_hmm import train_hmm
    
    train, test = load_pos_data()
    vocab, tags = get_vocab_and_tags(train)
    
    hmm_model = train_hmm(train, vocab, tags)
    mft_model = train_mft_baseline(train, tags)
    
    results = evaluate_models(test[:100], hmm_model, mft_model)
    print(f"HMM Accuracy:  {results['hmm_accuracy']:.4f}")
    print(f"MFT Accuracy:  {results['mft_accuracy']:.4f}")
