import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_processing import clean_reviews
from src.review_scoring import add_quality_scores
from src.review_analysis import add_sentiment, add_topic_clusters, reviewer_summary
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Apparel Review Intelligence", page_icon="👕", layout="wide")
st.title("Apparel Review Intelligence")
st.caption("A practical analytics workflow for review quality, reviewer performance and product feedback.")

@st.cache_data
def load_demo():
    return pd.read_csv("data/reviews.csv")

uploaded = st.sidebar.file_uploader("Upload your review CSV (optional)", type=["csv"])
raw = pd.read_csv(uploaded) if uploaded else load_demo()
st.sidebar.caption("Using the included demo dataset." if uploaded is None else "Using your uploaded dataset.")

try:
    df = clean_reviews(raw)
    df = add_quality_scores(df)
    df = add_sentiment(df)
    df = add_topic_clusters(df)
except Exception as exc:
    st.error(f"Could not process this dataset: {exc}")
    st.info("Required columns: reviewer_id, product_id, review_title, review_text, rating. Optional: helpful_votes, verified_purchase, review_date.")
    st.stop()

if "helpful_votes" not in df.columns:
    df["helpful_votes"] = 0
df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0)
if "verified_purchase" not in df.columns:
    df["verified_purchase"] = "Unknown"

# Filters
st.sidebar.subheader("Filters")
products = sorted(df["product_id"].astype(str).unique())
selected_products = st.sidebar.multiselect("Products", products, default=products)
bands = ["High", "Medium", "Low"]
selected_bands = st.sidebar.multiselect("Quality bands", bands, default=bands)
view = df[df["product_id"].astype(str).isin(selected_products) & df["quality_band"].isin(selected_bands)].copy()
if view.empty:
    st.warning("No reviews match the selected filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews analyzed", f"{len(view):,}")
c2.metric("Average quality score", f"{view.quality_score.mean():.1f}/100")
c3.metric("High-quality reviews", f"{(view.quality_band == 'High').mean()*100:.1f}%")
c4.metric("Average rating", f"{view.rating.mean():.2f}/5")

left, right = st.columns(2)
with left:
    st.subheader("Review quality distribution")
    counts = view["quality_band"].value_counts().reindex(bands, fill_value=0).reset_index()
    counts.columns = ["Quality band", "Reviews"]
    st.plotly_chart(px.bar(counts, x="Quality band", y="Reviews", color="Quality band"), use_container_width=True)
with right:
    st.subheader("Sentiment distribution")
    sent = view["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Reviews")
    st.plotly_chart(px.pie(sent, names="Sentiment", values="Reviews", hole=0.45), use_container_width=True)

st.subheader("Does positive sentiment mean a useful review?")
st.plotly_chart(px.box(view, x="sentiment", y="quality_score", color="sentiment",
                       labels={"quality_score":"Heuristic quality score", "sentiment":"Sentiment"}),
                use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Reviewer performance")
    reviewers = reviewer_summary(view).head(10)
    st.dataframe(reviewers.round(2), use_container_width=True, hide_index=True)
with right:
    st.subheader("Feedback topic clusters")

    cluster_counts = (
        view["topic_cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Reviews")
    )

    st.plotly_chart(
        px.bar(cluster_counts, x="Cluster", y="Reviews"),
        use_container_width=True
    )

    # Extract important keywords for each cluster
    st.markdown("### What are customers talking about?")

    try:
        texts = view["review"].fillna("").astype(str)

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=1000,
            ngram_range=(1, 2)
        )

        X = vectorizer.fit_transform(texts)
        terms = vectorizer.get_feature_names_out()

        topic_summary = []

        for cluster in sorted(view["topic_cluster"].dropna().unique()):

            positions = [
                i for i, value in enumerate(view["topic_cluster"].tolist())
                if value == cluster
            ]

            scores = X[positions].mean(axis=0).A1
            top_indices = scores.argsort()[-5:][::-1]

            keywords = ", ".join(terms[i] for i in top_indices)

            example = texts.iloc[positions[0]]

            topic_summary.append({
                "Cluster": int(cluster),
                "Reviews": len(positions),
                "Top keywords": keywords,
                "Example review": example
            })

        st.dataframe(
            pd.DataFrame(topic_summary),
            use_container_width=True,
            hide_index=True
        )

    except ValueError:
        st.warning("Not enough meaningful review text to extract keywords.")

st.subheader("Review-level data")
show_cols = ["reviewer_id", "product_id", "rating", "sentiment", "quality_score", "quality_band", "topic_cluster", "review"]
st.dataframe(view[show_cols], use_container_width=True, hide_index=True)
st.download_button("Download analyzed reviews CSV", view.to_csv(index=False).encode("utf-8"),
                   file_name="analyzed_apparel_reviews.csv", mime="text/csv")

with st.expander("Scoring methodology and limitations"):
    st.markdown("""
    The quality score is a transparent **heuristic**, not a scientifically validated label.
    It awards points for review detail (20), product attributes mentioned (30),
    practical usefulness (30), and contextual specificity (20). Sentiment is calculated
    separately using VADER. Therefore, negative feedback can still receive a high quality score.
    Topic clusters are unsupervised groups of similar text; they are not guaranteed to map
    perfectly to human-defined product themes. Validate scores and clusters with human reviewers
    before using them for operational decisions.
    """)
