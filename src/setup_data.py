import os
import requests
import zipfile
import io
import nltk

def manual_download_treebank():
    """
    Manually download and extract the NLTK treebank corpus to bypass downloader failures.
    """
    # NLTK default data path for user
    user_home = os.path.expanduser("~")
    nltk_data_dir = os.path.join(user_home, "nltk_data")
    corpora_dir = os.path.join(nltk_data_dir, "corpora")
    
    # Create directories if they don't exist
    os.makedirs(corpora_dir, exist_ok=True)
    
    print(f"Target directory: {corpora_dir}")
    
    # URLs for the required packages
    sources = {
        "treebank": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/treebank.zip",
        "universal_tagset": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/universal_tagset.zip"
    }
    
    for name, url in sources.items():
        print(f"\nDownloading {name}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            print(f"Extracting {name}...")
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # For tagsets, they go into a 'taggers' folder
                if name == "universal_tagset":
                    target = os.path.join(nltk_data_dir, "taggers")
                    os.makedirs(target, exist_ok=True)
                    z.extractall(target)
                else:
                    z.extractall(corpora_dir)
                    
            print(f"Successfully installed {name}.")
        except Exception as e:
            print(f"Failed to download {name}: {e}")
            print("\n" + "="*50)
            print(f"MANUAL DOWNLOAD REQUIRED FOR {name.upper()}:")
            print(f"1. Download: {url}")
            print(f"2. Extract the ZIP file.")
            print(f"3. Move the extracted folder to: {target if name == 'universal_tagset' else corpora_dir}")
            print("="*50 + "\n")

if __name__ == "__main__":
    manual_download_treebank()
    
    # Verify
    try:
        from nltk.corpus import treebank
        print("\nVerification: Loading treebank...")
        sents = treebank.tagged_sents(tagset='universal')
        print(f"Success! Loaded {len(sents)} sentences.")
    except Exception as e:
        print(f"\nVerification failed: {e}")
