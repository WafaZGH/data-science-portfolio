## Steam Games Analysis 

This project was part of my data science training.
The goal was to analyze the Steam gaming platform dataset, exploring trends in game releases, reviews, prices, and languages, and to create insights into what drives game popularity.

The Challenge

We worked with a large dataset of Steam games containing:

Game metadata → title, release date, publisher, platforms, genres

Pricing & discount information

User reviews (positive/negative counts)

Supported languages

The main objectives were to:

Clean and transform raw nested JSON data into structured tables.

Explore trends in releases, reviews, and pricing across genres and years.

Identify patterns in discounts and their relation to user engagement.

Visualize the most common languages and top-performing games.

Context

The pipeline combines:

Steam dataset (JSON format) → parsed into structured columns

Databricks / PySpark → to handle large-scale transformations and aggregations

Visualization libraries (Matplotlib, Plotly) → to highlight key insights

The project focuses on making sense of raw platform data and extracting useful business and market insights for the gaming industry.

My Approach
1. Data Preparation

Parsed nested JSON into a clean base DataFrame (df_base).

Selected key fields: game name, release date, publisher, genre, platforms, price, discount, reviews, age requirements.

Cleaned and normalized text fields (languages, genres).

2. Data Transformation

Created derived features:

Total reviews = positive + negative

Positive ratio = % of positive reviews

Has_discount = flag if discount > 0

Split and exploded language and genre lists into individual rows.

Aggregated counts to identify most frequent languages and genres.

3. Analysis & Visualization

Languages → Top languages supported across Steam games.

Reviews → Distribution of positive ratios and total reviews.

Pricing → Relation between discounts and reviews.

Trends → Releases by year, growth in publishers and platforms.


## Project Assets

- Notebook (Databricks): https://dbc-be649761-69e7.cloud.databricks.com/editor/notebooks/3059052065921878?o=4205469511372996
  
