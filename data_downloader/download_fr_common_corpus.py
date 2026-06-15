import pandas as pd
import os
from huggingface_hub import hf_hub_download

os.makedirs("Data - French/CommonCorpus", exist_ok=True)

output_file = "Data - French/CommonCorpus/common_corpus_fr.txt"
total = 0

# Télécharge et filtre les premiers shards
for i in range(1, 11): # common_corpus_1 à common_corpus_10
    for j in range(1, 10): # subbset_10_1 à subset_10_10
        try:
            path = hf_hub_download(
                repo_id="PleIAs/common_corpus",
                filename=f"common_corpus_{i}/subset_10_{j}.parquet",
                repo_type="dataset"
            )
            df = pd.read_parquet(path)
            fr = df[df["language"] == "French"]["text"]
            with open(output_file, "a", encoding="utf-8") as f:
                for text in fr:
                    f.write(text + "\n")
                    total += len(text)
            print(f"common_corpus_{i}/subset_10_{j} ===> {total/1e9:.2f} GB français")
            if total >= 10_000_000_000:
                break
        except Exception as e:
            print(f"Skip: {e}")