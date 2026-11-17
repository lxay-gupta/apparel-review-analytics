import re
import pandas as pd

ATTRIBUTE_TERMS = {
    "fit & sizing": ["fit", "size", "sleeve", "length", "waist", "loose", "tight", "small", "large"],
    "fabric & material": ["fabric", "material", "cotton", "polyester", "texture", "thin", "thick"],
    "comfort": ["comfortable", "comfort", "soft", "itchy", "breathable", "lightweight"],
    "quality & durability": ["quality", "durable", "stitch", "seam", "wash", "fade", "tear"],
    "appearance": ["color", "colour", "style", "design", "look", "pattern"]
}

def score_one_review(text: str) -> dict:
    """
    Transparent heuristic score (0-100), not a validated ground-truth quality label.
    Rewards specificity and actionable product detail, not positive sentiment.
    """
    t = str(text).lower()
    words = re.findall(r"\b\w+\b", t)
    word_count = len(words)
    attribute_hits = sum(1 for terms in ATTRIBUTE_TERMS.values() if any(term in t for term in terms))
    evidence_terms = ["because", "although", "but", "after", "when", "compared", "however", "while"]
    evidence_hits = sum(1 for term in evidence_terms if term in t)
    practical_terms = ["recommend", "return", "work", "runs", "shrunk", "washed", "ordered", "expected", "fit"]

    detail = min(word_count / 45, 1) * 20
    attributes = min(attribute_hits / 3, 1) * 30
    usefulness = min(sum(1 for term in practical_terms if term in t) / 3, 1) * 30
    specificity = min(evidence_hits / 2, 1) * 20
    total = round(detail + attributes + usefulness + specificity)

    if total >= 70:
        label = "High"
    elif total >= 40:
        label = "Medium"
    else:
        label = "Low"
    return {
        "quality_score": total,
        "quality_band": label,
        "attribute_categories_mentioned": attribute_hits
    }

def add_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    scores = df["review"].apply(score_one_review).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
