# Apparel Review Intelligence

A Python analytics project that structures apparel product reviews, estimates review usefulness, analyzes sentiment, groups recurring feedback themes and summarizes reviewer performance.

## Why this project?
The workflow demonstrates practical skills relevant to feedback research and data operations:
- organizing raw participant feedback into structured datasets;
- assessing review usefulness separately from positive/negative opinion;
- summarizing reviewer-level performance;
- applying NLP and unsupervised machine learning;
- communicating findings through an interactive dashboard.

## Features
- Data cleaning, validation, duplicate removal and feature creation
- Transparent 0–100 heuristic review-quality score
- VADER sentiment analysis
- TF-IDF + K-Means text clustering
- Reviewer performance summary
- Streamlit dashboard with filters, charts and CSV export

## Run locally

```bash
git clone <your-repository-url>
cd apparel-review-analytics
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

The included `data/reviews.csv` is a **small synthetic demonstration dataset** created to test the application. It is not real customer or participant research and should not be presented as such.

For a stronger portfolio submission, replace it with a properly licensed public apparel-review dataset (for example, the Women's E-Commerce Clothing Reviews dataset available through Kaggle). Keep the original data source and license/attribution in your report.

Required CSV columns:
- `reviewer_id`
- `product_id`
- `review_title`
- `review_text`
- `rating` (1–5)

Optional columns:
- `helpful_votes`
- `verified_purchase`
- `review_date`

Upload a CSV from the dashboard sidebar, or replace `data/reviews.csv` with a dataset using the required column names.

## Methodology

### Review quality score
The score is an interpretable heuristic, not a validated ground-truth measure:
- Review detail: up to 20 points
- Product attributes mentioned: up to 30 points
- Practical usefulness: up to 30 points
- Contextual specificity: up to 20 points

The score does not reward positive sentiment. A detailed negative review can score higher than a short positive review.

### Sentiment
VADER returns a compound sentiment score and a Positive, Neutral or Negative category. It is a general-purpose lexicon-based method and may misread sarcasm, domain-specific language or nuanced feedback.

### Topic discovery
TF-IDF converts review text into numerical features; K-Means groups similar reviews. Clusters need human interpretation and are not automatically meaningful product categories.

## Important limitations
- The demo data is synthetic.
- Heuristic quality scores need human validation before operational use.
- Reviewer rankings can be misleading when reviewers submit very different numbers of reviews.
- Sentiment and topic models can make mistakes.
- Do not use this project to make consequential decisions about individual reviewers without review, consent and appropriate validation.

## Suggested next steps
1. Test on a real, licensed apparel-review dataset.
2. Manually label a sample of reviews for quality and compare human judgments with heuristic scores.
3. Tune the scoring rubric with reviewer feedback.
4. Validate topic clusters and document representative examples.
