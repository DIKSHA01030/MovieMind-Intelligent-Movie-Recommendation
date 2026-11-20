
import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load datasets
movies = pd.read_csv("data/processed/movies_clean.csv")
ratings = pd.read_csv("data/processed/ratings_clean.csv")

print("Movies:", movies.shape)
print("Ratings:", ratings.shape)


# -------------------------------
# 1. CONTENT-BASED SCORES
# -------------------------------

movies["genres"] = movies["genres"].fillna("")

vectorizer = TfidfVectorizer(
    token_pattern=r"(?u)\b[\w-]+\b"
)

genre_matrix = vectorizer.fit_transform(
    movies["genres"].str.replace("|", " ", regex=False)
)

content_similarity = cosine_similarity(genre_matrix)

movie_id_to_index = pd.Series(
    movies.index,
    index=movies["movieId"]
).to_dict()


# -------------------------------
# 2. COLLABORATIVE SCORES
# -------------------------------

rating_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

user_means = rating_matrix.mean(axis=1)

centered_matrix = rating_matrix.sub(
    user_means,
    axis=0
)

user_similarity = cosine_similarity(
    centered_matrix.fillna(0)
)

similarity_df = pd.DataFrame(
    user_similarity,
    index=rating_matrix.index,
    columns=rating_matrix.index
)


# -------------------------------
# 3. POPULARITY SCORES
# -------------------------------

movie_stats = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
)

C = ratings["rating"].mean()
m = ratings.groupby("movieId").size().quantile(0.70)

movie_stats["weighted_rating"] = (
    (movie_stats["rating_count"] /
     (movie_stats["rating_count"] + m))
    * movie_stats["average_rating"]
    +
    (m / (movie_stats["rating_count"] + m)) * C
)

# Normalize weighted ratings
pop_min = movie_stats["weighted_rating"].min()
pop_max = movie_stats["weighted_rating"].max()

movie_stats["popularity_score"] = (
    (movie_stats["weighted_rating"] - pop_min)
    / (pop_max - pop_min + 1e-9)
)


# -------------------------------
# 4. HYBRID RECOMMENDATION
# -------------------------------

def recommend_hybrid(
    user_id,
    movie_title,
    top_n=10,
    alpha=0.50,
    beta=0.30,
    gamma=0.20
):

    if user_id not in rating_matrix.index:
        print("Invalid user ID!")
        return pd.DataFrame()

    selected = movies[
        movies["title"].str.lower() == movie_title.lower()
    ]

    if selected.empty:
        print("Movie not found!")
        return pd.DataFrame()

    selected_id = selected.iloc[0]["movieId"]
    selected_index = movie_id_to_index[selected_id]

    # Content scores
    content_scores = pd.Series(
        content_similarity[selected_index],
        index=movies["movieId"]
    )

    # Collaborative scores
    similar_users = similarity_df[user_id].drop(user_id)

    similar_users = similar_users.sort_values(
        ascending=False
    ).head(20)

    neighbor_ratings = rating_matrix.loc[
        similar_users.index
    ]

    neighbor_means = user_means.loc[similar_users.index]

    centered_neighbors = neighbor_ratings.sub(
        neighbor_means,
        axis=0
    )

    weights = similar_users.to_numpy()

    numerator = centered_neighbors.fillna(0).T.dot(weights)

    denominator = (
        neighbor_ratings.notna().T.dot(np.abs(weights))
    )

    predictions = (
        numerator / denominator.replace(0, np.nan)
    )

    predictions = predictions + user_means[user_id]

    predictions = predictions.clip(0.5, 5.0)

    # Normalize collaborative predictions
    collaborative_scores = (
        (predictions - 0.5) / 4.5
    ).clip(0, 1)

    # Combine scores
    result = pd.DataFrame({
        "movieId": movies["movieId"],
        "content_score": content_scores.reindex(
            movies["movieId"]
        ).fillna(0).to_numpy(),
        "collaborative_score": collaborative_scores.reindex(
            movies["movieId"]
        ).fillna(0).to_numpy(),
        "popularity_score": movie_stats[
            "popularity_score"
        ].reindex(movies["movieId"]).fillna(0).to_numpy()
    })

    # Weighted hybrid score
    result["hybrid_score"] = (
        alpha * result["content_score"]
        + beta * result["collaborative_score"]
        + gamma * result["popularity_score"]
    )

    # Remove selected movie and already-rated movies
    rated_movies = rating_matrix.loc[user_id].dropna().index

    result = result[
        ~result["movieId"].isin(rated_movies)
    ]

    result = result[
        result["movieId"] != selected_id
    ]

    # Add movie titles
    result = result.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    result = result.sort_values(
        "hybrid_score",
        ascending=False
    ).head(top_n)

    return result[
        [
            "movieId",
            "title",
            "genres",
            "content_score",
            "collaborative_score",
            "popularity_score",
            "hybrid_score"
        ]
    ]


# -------------------------------
# 5. TEST MODEL
# -------------------------------

if __name__ == "__main__":

    recommendations = recommend_hybrid(
        user_id=1,
        movie_title="Toy Story (1995)",
        top_n=10
    )

    print("\n========== IMPROVED HYBRID RESULTS ==========")

    print(recommendations.to_string(index=False))

    os.makedirs("data/processed", exist_ok=True)

    recommendations.to_csv(
        "data/processed/improved_hybrid_recommendations.csv",
        index=False
    )

    print("\nRecommendations saved successfully!")
    print("\nIMPROVED HYBRID MODEL COMPLETED!")
