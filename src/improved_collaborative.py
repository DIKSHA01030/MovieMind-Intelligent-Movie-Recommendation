
import pandas as pd
import numpy as np
import os

from sklearn.metrics.pairwise import cosine_similarity


# Load datasets
ratings = pd.read_csv("data/processed/ratings_clean.csv")
movies = pd.read_csv("data/processed/movies_clean.csv")

print("Ratings:", ratings.shape)
print("Movies:", movies.shape)


# Create user-item matrix
rating_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)


# Calculate user average ratings
user_means = rating_matrix.mean(axis=1)

# Mean-center ratings
centered_matrix = rating_matrix.sub(
    user_means,
    axis=0
)


# Calculate user similarity
# Missing values are handled as zero after centering
similarity_matrix = cosine_similarity(
    centered_matrix.fillna(0)
)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=rating_matrix.index,
    columns=rating_matrix.index
)


def recommend_for_user(user_id, top_n=10, k=20):

    if user_id not in rating_matrix.index:
        print("User ID not found!")
        return pd.DataFrame()

    # Find similar users
    similar_users = similarity_df[user_id].drop(user_id)

    similar_users = similar_users.sort_values(
        ascending=False
    ).head(k)

    # Ratings by selected neighbors
    neighbor_ratings = rating_matrix.loc[
        similar_users.index
    ]

    # Center each neighbor's ratings
    neighbor_means = user_means.loc[similar_users.index]

    centered_neighbors = neighbor_ratings.sub(
        neighbor_means,
        axis=0
    )

    # Weighted rating deviations
    weights = similar_users.to_numpy()

    numerator = centered_neighbors.fillna(0).T.dot(weights)

    denominator = (
        neighbor_ratings.notna().T.dot(
            np.abs(weights)
        )
    )

    # Predict rating
    predictions = (
        numerator / denominator.replace(0, np.nan)
    )

    predictions = predictions + user_means[user_id]

    predictions = predictions.clip(0.5, 5.0)

    # Remove already rated movies
    rated_movies = rating_matrix.loc[user_id].dropna().index

    predictions = predictions.drop(
        index=rated_movies,
        errors="ignore"
    )

    # Sort predictions
    recommendations = predictions.dropna().sort_values(
        ascending=False
    ).head(top_n)

    # Convert into DataFrame
    result = recommendations.rename(
        "predicted_rating"
    ).reset_index()

    result = result.merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left"
    )

    return result[
        ["movieId", "title", "predicted_rating"]
    ]


# Test recommendation
if __name__ == "__main__":

    user_id = 1

    print("\n========== IMPROVED RECOMMENDATIONS ==========")

    recommendations = recommend_for_user(
        user_id=user_id,
        top_n=10,
        k=20
    )

    print(recommendations.to_string(index=False))

    os.makedirs("data/processed", exist_ok=True)

    recommendations.to_csv(
        "data/processed/improved_recommendations.csv",
        index=False
    )

    print("\nRecommendations saved!")
    print("\nIMPROVED COLLABORATIVE FILTERING COMPLETED!")

