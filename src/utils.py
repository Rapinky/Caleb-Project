"""
Utility functions for HMM Project - POS Tagging.
"""

import nltk
from collections import Counter
from nltk.corpus import treebank
from sklearn.model_selection import train_test_split
import numpy as np

def download_nltk_data():
    """
    Download necessary NLTK data. Skip if connection fails.
    """
    print("Checking for NLTK data...")
    # These are small and might succeed or fail fast
    nltk.download('treebank', quiet=True, raise_on_error=False)
    nltk.download('universal_tagset', quiet=True, raise_on_error=False)

def generate_synthetic_data():
    """
    Generate a small synthetic POS dataset for testing when NLTK download fails.
    """
    print("WARNING: Using synthetic data fallback.")
    synthetic_sents = [
        # Det/Noun/Verb
        [("the", "DET"), ("dog", "NOUN"), ("runs", "VERB")],
        [("a", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")],
        [("the", "DET"), ("man", "NOUN"), ("walks", "VERB"), ("slowly", "ADV")],
        [("a", "DET"), ("woman", "NOUN"), ("talks", "VERB"), ("quickly", "ADV")],
        
        # Pronouns & State
        [("i", "PRON"), ("am", "VERB"), ("happy", "ADJ")],
        [("she", "PRON"), ("is", "VERB"), ("here", "ADV")],
        [("they", "PRON"), ("are", "VERB"), ("walking", "VERB")],
        [("it", "PRON"), ("was", "VERB"), ("small", "ADJ")],
        [("we", "PRON"), ("were", "VERB"), ("there", "ADV")],
        
        # Prepositions & Objects
        [("the", "DET"), ("book", "NOUN"), ("is", "VERB"), ("on", "ADP"), ("the", "DET"), ("table", "NOUN")],
        [("my", "DET"), ("friend", "NOUN"), ("is", "VERB"), ("in", "ADP"), ("the", "DET"), ("office", "NOUN")],
        [("she", "PRON"), ("put", "VERB"), ("the", "DET"), ("key", "NOUN"), ("on", "ADP"), ("my", "DET"), ("table", "NOUN")],
        
        # More complex
        [("i", "PRON"), ("see", "VERB"), ("a", "DET"), ("quick", "ADJ"), ("brown", "ADJ"), ("fox", "NOUN")],
        [("wow", "PRT"), ("!", "."), ("she", "PRON"), ("is", "VERB"), ("very", "ADV"), ("fast", "ADJ")],
        [("and", "CONJ"), ("i", "PRON"), ("was", "VERB"), ("very", "ADV"), ("happy", "ADJ")],
        
        # Variety
        [("the", "DET"), ("quick", "ADJ"), ("fox", "NOUN"), ("jumps", "VERB"), ("over", "ADP"), ("the", "DET"), ("lazy", "ADJ"), ("dog", "NOUN")],
        [("who", "PRON"), ("is", "VERB"), ("that", "DET"), ("man", "NOUN")],
        [("this", "DET"), ("is", "VERB"), ("a", "DET"), ("test", "NOUN")],
    ]
    # Repeat significantly to ensure a robust transition matrix
    return synthetic_sents * 50

def load_pos_data(test_size=0.2, random_state=42, use_synthetic=False):
    """
    Load the NLTK treebank corpus and split into train/test sets.
    If NLTK load fails or use_synthetic is True, falls back to synthetic data.
    """
    if use_synthetic:
        sentences = generate_synthetic_data()
    else:
        try:
            # Use universal tagset for simplicity
            sentences = treebank.tagged_sents(tagset='universal')
            print(f"Loaded {len(sentences)} sentences from NLTK treebank.")
        except Exception as e:
            print(f"Could not load NLTK treebank: {e}")
            sentences = generate_synthetic_data()
    
    train_data, test_data = train_test_split(
        sentences, test_size=test_size, random_state=random_state
    )
    
    return train_data, test_data

def get_vocab_and_tags(train_data, min_freq=1):
    """
    Extract unique vocabulary and tags from training data.
    Words with frequency < min_freq are mapped to <UNK>.
    """
    word_counts = Counter()
    tags = set()
    
    for sentence in train_data:
        for word, tag in sentence:
            word_counts[word.lower()] += 1
            tags.add(tag)
            
    # Filter vocab by frequency
    vocab = set(['<UNK>'])
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab.add(word)
            
    return sorted(list(vocab)), sorted(list(tags))

if __name__ == "__main__":
    download_nltk_data()
    train, test = load_pos_data()
    vocab, tags = get_vocab_and_tags(train)
    print(f"Training sentences: {len(train)}")
    print(f"Test sentences: {len(test)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of tags: {len(tags)} ({tags})")
