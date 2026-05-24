import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# LOAD DATA
# =========================

df = pd.read_csv('amazon_products_sales_data_uncleaned.csv')

print(df.head())


# =========================
# CLEAN NUMBER OF REVIEWS
# =========================

df['number_of_reviews'] = (
    df['number_of_reviews']
    .astype(str)
    .str.replace(',', '')
)

df['number_of_reviews'] = pd.to_numeric(
    df['number_of_reviews'],
    errors='coerce'
)

df['number_of_reviews'] = (
    df['number_of_reviews']
    .fillna(0)
)

print(df['number_of_reviews'].head())


# =========================
# CLEAN LISTED PRICE
# =========================

df['listed_price'] = (
    df['listed_price']
    .astype(str)
    .replace(r'[\$,]', '', regex=True)
)

df['listed_price'] = pd.to_numeric(
    df['listed_price'],
    errors='coerce'
)

df['listed_price'] = (
    df['listed_price']
    .fillna(df['listed_price'].median())
)

print(df['listed_price'].head())


# =========================
# CLEAN BOUGHT IN LAST MONTH
# =========================

def convert_bought(values):

    if pd.isna(values):
        return 0

    values = str(values)

    # Handle K values
    if 'K' in values:
        num = float(values.split('K')[0])
        return num * 1000

    digits = re.findall(r'\d+', values)

    return float(digits[0]) if digits else 0


df['bought_in_last_month'] = (
    df['bought_in_last_month']
    .apply(convert_bought)
)

print(df['bought_in_last_month'].head())


# =========================
# CLEAN BEST SELLER FEATURE
# =========================

df['is_best_seller_clean'] = (
    df['is_best_seller']
    .astype(str)
    .str.contains(
        'Best Seller',
        case=False,
        na=False
    )
    .astype(int)
)

print(df['is_best_seller_clean'].value_counts())


# =========================
# CLEAN TITLES
# =========================

def clean_title(title):

    title = str(title).lower()

    # Remove special characters
    title = re.sub(
        r'[^a-zA-Z0-9\s]',
        '',
        title
    )

    # Remove extra spaces
    title = re.sub(
        r'\s+',
        ' ',
        title
    )

    return title.strip()


df['clean_title'] = (
    df['title']
    .apply(clean_title)
)

print(df['clean_title'].head())


# =========================
# TF-IDF VECTORIZATION
# =========================

tfidf = TfidfVectorizer(
    stop_words='english'
)

tfidf_matrix = tfidf.fit_transform(
    df['clean_title']
)

print(tfidf_matrix)


# =========================
# COSINE SIMILARITY
# =========================

cosine_sim = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)

print(cosine_sim.shape)


# =========================
# RECOMMENDATION FUNCTION   
# =========================

def recommend_products(product_title, top_n=5):

    product_title = clean_title(product_title)

    # Find matching products
    matches = df[
        df['clean_title']
        .str.contains(product_title, na=False)
    ]

    if matches.empty:
        print("Product not found")
        return

    idx = matches.index[0]

    # Similarity scores
    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    # Sort by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove same product
    similarity_scores = similarity_scores[1:top_n+1]

    print("\nRecommended Products:\n")

    for i in similarity_scores:

        product_index = i[0]
        similarity_score = i[1]

        print("Title:",
              df.iloc[product_index]['title'])

        print("Similarity Score:",
              round(similarity_score, 3))

        print("Price:",
              df.iloc[product_index]['listed_price'])

        print("Reviews:",
              df.iloc[product_index]['number_of_reviews'])

        print("-" * 60)


# =========================
# TEST RECOMMENDATION
# =========================

recommend_products("laptops")