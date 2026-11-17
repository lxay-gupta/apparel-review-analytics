import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    out = df.copy()
    out["sentiment_compound"] = out["review"].apply(
        lambda text: analyzer.polarity_scores(str(text))["compound"]
    )
    out["sentiment"] = out["sentiment_compound"].apply(
        lambda score: "Positive" if score >= 0.05 else ("Negative" if score <= -0.05 else "Neutral")
    )
    return out

def add_topic_clusters(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    out = df.copy()
    texts = out["review"].fillna("").astype(str)
    if len(out) < 3 or texts.str.strip().eq("").all():
        out["topic_cluster"] = 0
        return out
    k = max(2, min(int(n_clusters), len(out) - 1))
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1500, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    out["topic_cluster"] = model.fit_predict(matrix)
    return out

def reviewer_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("reviewer_id", as_index=False).agg(
        reviews_submitted=("review", "count"),
        average_quality=("quality_score", "mean"),
        high_quality_share=("quality_band", lambda x: (x == "High").mean() * 100),
        average_rating=("rating", "mean"),
        average_helpful_votes=("helpful_votes", "mean")
    )
    return summary.sort_values(["average_quality", "reviews_submitted"], ascending=[False, False])
