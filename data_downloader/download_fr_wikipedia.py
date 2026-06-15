from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train")

with open("Data - French/Wikipedia/wikipedia_fr.txt", "w", encoding="utf-8") as f:
    for i, article in enumerate(ds):
        f.write(article["text"] + "\n")