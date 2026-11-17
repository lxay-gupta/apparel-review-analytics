import re
import pandas as pd

REQUIRED_COLUMNS = ["reviewer_id", "product_id", "review_title", "review_text", "rating"]

def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, normalize and enrich raw review records."""
    df = df.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    for col in ["reviewer_id", "product_id", "review_title", "review_text"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df = df[df["rating"].between(1, 5)]
    df = df.drop_duplicates(subset=["reviewer_id", "product_id", "review_text"])
    df["review"] = (df["review_title"] + ". " + df["review_text"]).str.strip()
    df["review"] = df["review"].str.replace(r"\s+", " ", regex=True)
    df["word_count"] = df["review"].apply(lambda x: len(re.findall(r"\b\w+\b", x)))
    df["char_count"] = df["review"].str.len()
    return df.reset_index(drop=True)
