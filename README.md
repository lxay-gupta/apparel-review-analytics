
# Apparel Review Intelligence

**An interactive customer feedback analytics platform built with Python, Machine Learning and Streamlit.**

Apparel Review Intelligence processes customer product reviews and converts unstructured textual feedback into structured business insights. The application combines review quality scoring, sentiment analysis, text-based clustering and reviewer analytics in an interactive dashboard.

Users can analyse the included sample dataset or upload their own review data in CSV format.

---

## 1. Project Overview

Customer reviews contain valuable information about product quality, customer satisfaction, sizing, comfort, durability and purchasing experiences. However, analysing large volumes of unstructured feedback manually can be time-consuming.

This project implements a data analytics pipeline to:

- Clean and prepare customer review data.
- Evaluate the informational quality of individual reviews.
- Classify customer sentiment.
- Group similar reviews using unsupervised machine learning.
- Analyse reviewer-level contributions.
- Present analytical results through an interactive dashboard.

The application is designed around apparel product feedback, but its CSV-based workflow can also be adapted to other product review datasets.

---

## 2. Key Features

### Review Data Processing
- CSV dataset ingestion using Pandas.
- Data cleaning and standardisation.
- Handling of missing and duplicate records.
- Preparation of review text and numerical fields for analysis.

### Review Quality Scoring
- Calculates a heuristic quality score for individual reviews.
- Evaluates review detail, product attributes, practical usefulness and contextual specificity.
- Assigns High, Medium and Low quality categories.
- Enables filtering based on review quality.

### Sentiment Analysis
- Uses the VADER sentiment analysis model.
- Evaluates the polarity of customer review text.
- Categorises reviews as Positive, Neutral or Negative.
- Presents sentiment distribution through interactive visualisations.

### Feedback Topic Clustering
- Converts textual reviews into numerical feature representations using TF-IDF.
- Applies K-Means clustering to group reviews with similar textual patterns.
- Identifies groups of related customer feedback.
- Displays cluster-level review counts and keywords for topic interpretation.

### Reviewer Analytics
- Aggregates review activity by reviewer ID.
- Calculates review counts and average quality scores.
- Analyses high-quality review proportions and average ratings.
- Provides reviewer-level summary tables.

### Interactive Analytics Dashboard
- Built using Streamlit.
- Displays key performance indicators and interactive charts.
- Supports product and review quality filters.
- Allows users to upload custom CSV datasets.
- Enables downloading processed review data.

---

## 3. Technical Architecture

The application follows a modular Python architecture, separating data processing, scoring, text analysis and dashboard presentation.

```text
                  Customer Review Dataset
                           |
                           v
                  CSV Data Ingestion
                           |
                           v
                  Data Preprocessing
                  (Pandas)
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
        Quality Score  Sentiment     Text Feature
         Calculation   Analysis      Extraction
             |        (VADER)        (TF-IDF)
             |             |             |
             |             |             v
             |             |        K-Means
             |             |        Clustering
             |             |             |
             +-------------+-------------+
                           |
                           v
                  Integrated Analysis
                           |
                           v
                 Streamlit Dashboard
                           |
              +------------+------------+
              |            |            |
              v            v            v
          Metrics       Charts       Data Export
```

### Processing Modules

| Module | Responsibility |
|---|---|
| `data_processing.py` | Data cleaning and preprocessing |
| `review_scoring.py` | Review quality score calculation |
| `review_analysis.py` | Sentiment analysis, topic clustering and reviewer summaries |
| `app.py` | Dashboard interface, filters, visualisations and CSV export |

---

## 4. Methodology and Algorithms

### 4.1 Data Preprocessing

Pandas is used to load and manipulate the review dataset.

The preprocessing stage prepares the input data for downstream analysis by cleaning review records, handling missing values and removing duplicate entries.

The cleaned dataset is passed to the scoring and analysis modules.

### 4.2 Review Quality Scoring

A rule-based heuristic scoring mechanism evaluates the informational value of each review.

The score is calculated using four components:

| Component | Maximum Score |
|---|---:|
| Review detail | 20 |
| Product attributes mentioned | 30 |
| Practical usefulness | 30 |
| Contextual specificity | 20 |
| **Total** | **100** |

The resulting score is used to classify reviews:

| Quality Category | Score Range |
|---|---:|
| High | 70–100 |
| Medium | 40–69 |
| Low | Below 40 |

This scoring mechanism distinguishes informational value from customer sentiment. For example, a negative review describing a specific stitching defect can receive a high quality score because it provides actionable product feedback.

### 4.3 Sentiment Analysis using VADER

VADER (Valence Aware Dictionary and sEntiment Reasoner) is used to analyse the emotional polarity of review text.

The model evaluates textual sentiment and assigns each review to one of three categories:

- Positive
- Neutral
- Negative

Sentiment is analysed independently of the review quality score. This allows the application to distinguish between customer satisfaction and the usefulness of customer feedback.

### 4.4 Text Feature Extraction using TF-IDF

Term Frequency–Inverse Document Frequency (TF-IDF) converts textual review data into numerical feature vectors.

The technique assigns importance to words based on their frequency within a review and their distribution across the document collection.

These numerical representations provide input features for the clustering algorithm.

### 4.5 Unsupervised Learning using K-Means

K-Means clustering groups reviews based on similarities in their TF-IDF feature representations.

The algorithm partitions the review dataset into five clusters.

Each cluster represents a group of reviews with similar textual patterns. Cluster-level keyword extraction helps interpret the underlying feedback themes.

The resulting groups can help identify recurring discussion areas such as material quality, sizing, comfort and durability.

### 4.6 Reviewer-Level Aggregation

Reviewer analytics are generated by grouping records using reviewer identifiers.

The application calculates metrics such as:

- Total reviews submitted.
- Average review quality score.
- Proportion of high-quality reviews.
- Average product rating.
- Helpful vote information, when available.

These aggregated metrics provide a structured overview of reviewer contributions.

---

## 5. Dashboard Metrics and Visualisations

The Streamlit dashboard presents the following analytical outputs:

| Dashboard Component | Analytical Purpose |
|---|---|
| Reviews Analyzed | Total number of processed reviews |
| Average Quality Score | Mean review quality score |
| High-Quality Reviews | Percentage of reviews classified as High |
| Average Rating | Mean customer product rating |
| Review Quality Distribution | Distribution across quality categories |
| Sentiment Distribution | Positive, Neutral and Negative review proportions |
| Sentiment vs. Review Quality | Comparison of sentiment categories against quality scores |
| Reviewer Performance | Reviewer-level analytical summary |
| Feedback Topic Clusters | Distribution of reviews across K-Means clusters |
| Review-Level Data | Processed review records and analytical labels |

Plotly is used to generate interactive charts, while Streamlit provides the dashboard layout, filters and data export functionality.

---

## 6. Technology Stack

| Technology | Application |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF vectorisation and K-Means clustering |
| NLTK / VADER | Sentiment analysis |
| Streamlit | Interactive dashboard development |
| Plotly Express | Data visualisation |

---

## 7. Project Structure

```text
apparel-review-analytics/
│
├── app.py
│
├── data/
│   └── reviews.csv
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── review_scoring.py
│   └── review_analysis.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 8. Dataset Requirements

The application accepts CSV files containing customer product review information.

### Required Columns

| Column | Data Type | Description |
|---|---|---|
| reviewer_id | String | Unique reviewer identifier |
| product_id | String | Product identifier |
| review_title | String | Review headline |
| review_text | String | Full customer review |
| rating | Numeric | Product rating |

### Optional Columns

| Column | Description |
|---|---|
| helpful_votes | Number of helpful votes received |
| verified_purchase | Purchase verification status |
| review_date | Date of review submission |

The repository includes a synthetic sample dataset for demonstrating the application's functionality.

---

## 9. Installation and Execution

### Prerequisites

- Python 3.9 or later
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/apparel-review-analytics.git
cd apparel-review-analytics
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv .venv
```

### Step 3: Activate the Environment

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Launch the Application

```bash
streamlit run app.py
```

The dashboard will be available locally at:

```text
http://localhost:8501
```

---

## 10. Running the Application

1. Launch the Streamlit application.
2. Use the included sample dataset or upload a custom CSV file.
3. Apply product and review quality filters.
4. Explore review quality and sentiment distributions.
5. Examine feedback clusters and reviewer analytics.
6. Inspect individual processed reviews.
7. Export the analysed dataset as a CSV file.

---

## Project Information

**Project:** Apparel Review Intelligence  
**Category:** Data Analytics / Natural Language Processing / Machine Learning  
**Language:** Python  
**Interface:** Streamlit  
**Core Techniques:** Data Preprocessing, Heuristic Scoring, Sentiment Analysis, TF-IDF and K-Means Clustering
