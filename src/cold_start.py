
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# LOAD DATA
# ==========================================

MOVIES_PATH = "data/processed/movies_clean.csv"
RATINGS_PATH = "data/processed/ratings_clean.csv"

movies = pd.read_csv(MOVIES_PATH)
ratings = pd.read_csv(RATINGS_PATH)

movies["genres"] = movies["genres"].fillna("")
movies["genres_text"] = movies["genres"].str.replace("|", " ", regex=False)


# ==========================================
# POPULARITY SCORE
# ==========================================

movie_stats = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()

C = ratings["rating"].mean()
m = movie_stats["rating_count"].quantile(0.70)

movie_stats["weighted_rating"] = (
    (movie_stats["rating_count"] / (movie_stats["rating_count"] + m))
    * movie_stats["average_rating"]
    +
    (m / (movie_stats["rating_count"] + m)) * C
)

movie_data = movies.merge(movie_stats, on="movieId", how="left")

movie_data["weighted_rating"] = movie_data["weighted_rating"].fillna(C)
movie_data["rating_count"] = movie_data["rating_count"].fillna(0)


# ==========================================
# CONTENT SIMILARITY
# ==========================================

vectorizer = TfidfVectorizer(
    token_pattern=r"(?u)\b[\w-]+\b"
)

genre_matrix = vectorizer.fit_transform(movie_data["genres_text"])

similarity_matrix = cosine_similarity(genre_matrix)


# ==========================================
# CASE 1: NEW USER
# ==========================================

def recommend_for_new_user(preferred_genres, top_n=10):

    preferred_genres = [
        genre.strip().lower()
        for genre in preferred_genres
    ]

    def genre_match(genres):
        movie_genres = [
            genre.strip().lower()
            for genre in genres.split("|")
        ]

        return sum(
            genre in movie_genres
            for genre in preferred_genres
        )

    recommendations = movie_data.copy()

    recommendations["genre_match"] = (
        recommendations["genres"].apply(genre_match)
    )

    recommendations = recommendations.sort_values(
        by=["genre_match", "weighted_rating", "rating_count"],
        ascending=False
    )

    return recommendations[
        ["title", "genres", "weighted_rating", "rating_count"]
    ].head(top_n)


# ==========================================
# CASE 2: NEW MOVIE
# ==========================================

def recommend_for_new_movie(movie_title, top_n=10):

    matches = movie_data[
        movie_data["title"].str.lower() == movie_title.lower()
    ]

    if matches.empty:
        print("Movie not found!")
        return pd.DataFrame()

    movie_index = matches.index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similar_indices = [
        index for index, score in similarity_scores
        if index != movie_index
    ][:top_n]

    recommendations = movie_data.iloc[similar_indices].copy()

    recommendations["similarity_score"] = [
        similarity_matrix[movie_index, index]
        for index in similar_indices
    ]

    return recommendations[
        ["title", "genres", "similarity_score"]
    ]


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":

    print("\n========== NEW USER RECOMMENDATIONS ==========")

    new_user_recommendations = recommend_for_new_user(
        ["Action", "Adventure", "Sci-Fi"],
        10
    )

    print(new_user_recommendations.to_string(index=False))


    print("\n========== NEW MOVIE RECOMMENDATIONS ==========")

    new_movie_recommendations = recommend_for_new_movie(
        "Toy Story (1995)",
        10
    )

    print(new_movie_recommendations.to_string(index=False))

    print("\nCOLD-START MODULE COMPLETED!")
