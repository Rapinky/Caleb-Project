"""
Module for training HMM POS Tagger.
"""

import numpy as np
from collections import defaultdict, Counter

def train_hmm(train_data, vocab, tags):
    """
    Train HMM parameters: Transition (A), Emission (B), and Initial (Pi).
    Uses Add-1 (Laplacian) smoothing.
    """
    tag2idx = {tag: i for i, tag in enumerate(tags)}
    word2idx = {word: i for i, word in enumerate(vocab)}
    
    num_tags = len(tags)
    num_words = len(vocab)
    
    # Initialize matrices with 1 for Laplacian smoothing
    A = np.ones((num_tags, num_tags))
    B = np.ones((num_tags, num_words))
    Pi = np.ones(num_tags)
    
    # Count occurrences
    for sentence in train_data:
        for i, (word, tag) in enumerate(sentence):
            t_idx = tag2idx[tag]
            
            # Initial state counts
            if i == 0:
                Pi[t_idx] += 1
            else:
                # Transition counts
                prev_tag = sentence[i-1][1]
                prev_t_idx = tag2idx[prev_tag]
                A[prev_t_idx, t_idx] += 1
            
            # Emission counts
            w_idx = word2idx.get(word.lower(), word2idx['<UNK>'])
            B[t_idx, w_idx] += 1
                
    # Normalize to probabilities
    # Divide each row by its sum
    A = A / A.sum(axis=1, keepdims=True)
    B = B / B.sum(axis=1, keepdims=True)
    Pi = Pi / Pi.sum()
    
    return {
        "A": A,
        "B": B,
        "Pi": Pi,
        "tag2idx": tag2idx,
        "word2idx": word2idx,
        "idx2tag": {i: tag for tag, i in tag2idx.items()}
    }

if __name__ == "__main__":
    from utils import load_pos_data, get_vocab_and_tags
    print("Loading data...")
    train, _ = load_pos_data()
    vocab, tags = get_vocab_and_tags(train)
    print("Training HMM...")
    model = train_hmm(train, vocab, tags)
    print("HMM Training Complete.")
    print(f"Transition Matrix Shape: {model['A'].shape}")
    print(f"Emission Matrix Shape: {model['B'].shape}")
