"""
Preprocessing for IMDB sentiment dataset.

Produces TWO versions of the text:
  - clean_text : lowercased, HTML/punctuation stripped -> for the classic TF-IDF baseline
  - text       : raw (only HTML tags removed)          -> for the DistilBERT fine-tune later

Transformers tokenizers work better on natural, cased text with punctuation intact,
so we deliberately do NOT lowercase/strip punctuation for that stage.
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/IMDB_Dataset.csv"
OUT_DIR = "data/processed"


def strip_html(text: str) -> str:
    return re.sub(r"<.*?>", " ", text)


def clean_for_classic(text: str) -> str:
    text = strip_html(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows")

    before = len(df)
    df = df.drop_duplicates(subset="review").reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate reviews -> {len(df)} remain")

    df = df.dropna(subset=["review", "sentiment"]).reset_index(drop=True)

    # label -> int
    df["label"] = (df["sentiment"] == "positive").astype(int)

    # raw text for transformer stage (HTML stripped only)
    df["text"] = df["review"].apply(strip_html).str.strip()

    # cleaned text for classic ML stage
    df["clean_text"] = df["review"].apply(clean_for_classic)

    # drop anything that became empty/too short after cleaning
    before = len(df)
    df = df[df["clean_text"].str.len() > 5].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows that were empty after cleaning")

    train_df, test_df = train_test_split(
        df[["text", "clean_text", "label"]],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

    print(f"\nTrain: {len(train_df)} rows, label balance:\n{train_df['label'].value_counts(normalize=True)}")
    print(f"\nTest: {len(test_df)} rows, label balance:\n{test_df['label'].value_counts(normalize=True)}")
    print("\nSample cleaned row:")
    print(train_df.iloc[0]["clean_text"][:300])


if __name__ == "__main__":
    main()