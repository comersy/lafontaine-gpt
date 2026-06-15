"""
Downloads all French books from Project Gutenberg via HuggingFace
and cleans them by removing headers and footers.

Output: Data - French/Gutenberg/gutenberg_fr.txt
"""

import os
import re
from datasets import load_dataset


# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = "Data - French/Gutenberg"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gutenberg_fr.txt")


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_gutenberg(text: str) -> str:
    """
    Removes Gutenberg header and footer.
    Keeps only the text between *** START OF *** and *** END OF ***.
    """
    # Remove header
    start = re.search(r"\*{3}\s*START OF[^\*]*\*{3}", text, re.IGNORECASE)
    if start:
        text = text[start.end():]

    # Remove footer
    end = re.search(r"\*{3}\s*END OF[^\*]*\*{3}", text, re.IGNORECASE)
    if end:
        text = text[:end.start()]

    # Remove lines with "Project Gutenberg" references
    lines = [
        line for line in text.splitlines()
        if "gutenberg" not in line.lower()
        and "produced by" not in line.lower()
        and "transcribed by" not in line.lower()
    ]

    return "\n".join(lines).strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading French Gutenberg dataset...")
    ds = load_dataset("manu/project_gutenberg", split="fr", streaming=True)

    total_chars = 0
    skipped     = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, book in enumerate(ds):
            text = clean_gutenberg(book["text"])

            if len(text) < 1000:
                skipped += 1
                continue

            f.write(text + "\n\n")
            total_chars += len(text)

            if i % 100 == 0:
                f.flush()
                print(f"  {i} books processed ===> {total_chars / 1_000_000:.1f} MB written")

    print(f"\nDone ===> {OUTPUT_FILE}")
    print(f"  Total: {total_chars / 1_000_000:.1f} MB")
    print(f"  Skipped: {skipped} books (too short after cleaning)")


if __name__ == "__main__":
    main()