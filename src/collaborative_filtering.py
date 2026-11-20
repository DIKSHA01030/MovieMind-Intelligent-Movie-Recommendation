
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

print("\n================================")
print("       MOVIEMIND PROJECT")
print("   COLLABORATIVE FILTERING")
print("================================\n")

# Load datasets
ratings = pd.read_csv("data/processed/ratings_clean.csv")
movies = pd.read_csv("data/processed/movies_clean.csv")

# Create User-Movie Rating Matrix
rating_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

print("USER-MOVIE MATRIX CREATED!")
print("Matrix Shape:", rating_matrix.shape)

# Replace missing ratings with 0 for similarity calculation
filled_matrix = rating_matrix.fillna(0)

# Calculate similarity between users
user_similarity = cosine_similarity(filled_matrix)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=rating_matrix.index,
    columns=rating_matrix.index
)

print("USER SIMILARITY MATRIX CREATED!")


# Recommendation function
def recommend_for_user(user_id, number_of_recommendations=10):

    if user_id not in rating_matrix.index:
        print("\nUser ID not found!")
        return

    # Find similar users
    similar_users = user_similarity_df.loc[user_id].drop(user_id)

    similar_users = similar_users.sort_values(
        ascending=False
    ).head(10)

    print("\nSimilar users for User", user_id)
    print(similar_users)

    # Get ratings of similar users
    neighbor_ratings = rating_matrix.loc[similar_users.index]

    # Calculate similarity-weighted ratings
    weighted_sum = neighbor_ratings.mul(
        similar_users, axis=0
    ).sum(axis=0)

    weight_sum = (
        neighbor_ratings.notna()
        .mul(similar_users, axis=0)
        .sum(axis=0)
    )

    recommendation_scores = weighted_sum.div(
        weight_sum.replace(0, float("nan"))
    )

    # Remove movies already rated by the target user
    already_rated = rating_matrix.loc[user_id].dropna().index

    recommendation_scores = recommendation_scores.drop(
        index=already_rated,
        errors="ignore"
    )

    # Sort recommendations
    recommendation_scores = recommendation_scores.dropna()
    recommendation_scores = recommendation_scores.sort_values(
        ascending=False
    ).head(number_of_recommendations)

    # Convert movie IDs into movie titles
    recommendations = recommendation_scores.rename(
        "predicted_rating"
    ).reset_index()

    recommendations = recommendations.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId"
    )

    recommendations["predicted_rating"] = (
        recommendations["predicted_rating"].round(2)
    )

    print("\nRecommended Movies for User:", user_id)
    print(
        recommendations[
            ["title", "genres", "predicted_rating"]
        ].to_string(index=False)
    )

    return recommendations


print("\nCOLLABORATIVE FILTERING READY!")

# Test recommendation system
recommend_for_user(1, 10)
