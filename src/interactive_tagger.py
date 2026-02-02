"""
Interactive Command Line Interface for the HMM POS Tagger.
"""

import sys
import os

# Add src to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_pos_data, get_vocab_and_tags, download_nltk_data
from train_hmm import train_hmm
from evaluate import viterbi

def main():
    print("=" * 50)
    print("   HMM POS Tagger: Interactive Session")
    print("=" * 50)
    
    # Setup
    print("Initializing system (loading data & training model)...")
    train_data, _ = load_pos_data()
    vocab, tags = get_vocab_and_tags(train_data, min_freq=2)
    model = train_hmm(train_data, vocab, tags)
    print("System ready!")
    print("-" * 50)
    
    while True:
        try:
            line = input("\nEnter a sentence (or 'q' to quit): ").strip()
            if not line:
                continue
            if line.lower() == 'q':
                print("Exiting. Goodbye!")
                break
                
            # Tokenize by whitespace
            words = line.split()
            if not words:
                continue
                
            # Predict
            tags_pred = viterbi(words, model)
            
            # Print results
            print("\nPredictions:")
            print(f"{'Word':15} | {'Tag'}")
            print("-" * 25)
            for w, t in zip(words, tags_pred):
                print(f"{w:15} | {t}")
                
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
