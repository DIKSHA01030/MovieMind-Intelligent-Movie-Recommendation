
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

print("\n================================")
print("       MOVIEMIND PROJECT")
print("    MODEL EVALUATION")
print("================================\n")

# 1. Load ratings
ratings = pd.read_csv("data/processed/ratings_clean.csv")

# 2. Split data into training and testing
train_data, test_data = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

print("Training records:", len(train_data))
print("Testing records:", len(test_data))

# 3. Create training user-movie matrix
train_matrix = train_data.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

# 4. Calculate user similarity using training data only
filled_matrix = train_matrix.fillna(0)

similarity = cosine_similarity(filled_matrix)

similarity_df = pd.DataFrame(
    similarity,
    index=train_matrix.index,
    columns=train_matrix.index
)

global_mean = train_data["rating"].mean()

# 5. Predict an individual rating
def predict_rating(user_id, movie_id, k=10):

    if user_id not in train_matrix.index:
        return global_mean

    if movie_id not in train_matrix.columns:
        return global_mean

    # Find similar users
    similar_users = (
        similarity_df.loc[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
    )

    # Keep users who rated this movie
    movie_ratings = train_matrix[movie_id].dropna()

    neighbors = similar_users[
        similar_users.index.isin(movie_ratings.index)
    ].head(k)

    if len(neighbors) == 0:
        user_mean = train_matrix.loc[user_id].mean()
        return user_mean if pd.notna(user_mean) else global_mean

    # Similarity-weighted rating
    neighbor_values = movie_ratings.loc[neighbors.index]

    denominator = neighbors.sum()

    if denominator <= 0:
        return global_mean

    prediction = np.dot(
        neighbors.values,
        neighbor_values.values
    ) / denominator

    return float(np.clip(prediction, 0.5, 5.0))


# 6. Predict test ratings
print("\nCalculating rating predictions...")

actual_ratings = []
predicted_ratings = []

for row in test_data.itertuples(index=False):

    prediction = predict_rating(
        row.userId,
        row.movieId
    )

    actual_ratings.append(row.rating)
    predicted_ratings.append(prediction)

# 7. Calculate MAE and RMSE
mae = mean_absolute_error(
    actual_ratings,
    predicted_ratings
)

rmse = np.sqrt(
    mean_squared_error(
        actual_ratings,
        predicted_ratings
    )
)

print("\n========== RATING EVALUATION ==========")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))


# 8. Precision@K and Recall@K
def evaluate_top_k(k=10, relevant_threshold=4.0):

    precision_values = []
    recall_values = []

    # Only evaluate users present in training data
    test_users = test_data[
        test_data["userId"].isin(train_matrix.index)
    ]["userId"].unique()

    for user_id in test_users:

        # Relevant movies in test data
        relevant_movies = set(
            test_data[
                (test_data["userId"] == user_id)
                & (test_data["rating"] >= relevant_threshold)
            ]["movieId"]
        )

        if len(relevant_movies) == 0:
            continue

        # Find similar users
        similar_users = (
            similarity_df.loc[user_id]
            .drop(user_id)
            .sort_values(ascending=False)
            .head(10)
        )

        neighbor_ratings = train_matrix.loc[similar_users.index]

        # Calculate weighted predicted ratings for all movies
        weighted_sum = neighbor_ratings.fillna(0).T.dot(
            similar_users
        )

        weight_sum = neighbor_ratings.notna().T.dot(
            similar_users
        )

        scores = weighted_sum.div(
            weight_sum.replace(0, np.nan)
        )

        # Exclude movies already rated by this user
        already_rated = train_matrix.loc[user_id].dropna().index
        scores = scores.drop(index=already_rated, errors="ignore")
        scores = scores.dropna()

        top_movies = set(
            scores.sort_values(ascending=False).head(k).index
        )

        hits = len(top_movies.intersection(relevant_movies))

        precision_values.append(hits / k)
        recall_values.append(hits / len(relevant_movies))

    if not precision_values:
        return None, None

    return (
        np.mean(precision_values),
        np.mean(recall_values)
    )


precision, recall = evaluate_top_k(k=10)

print("\n========== TOP-K EVALUATION ==========")

if precision is not None:
    print("Precision@10:", round(precision, 4))
    print("Recall@10:", round(recall, 4))
else:
    print("Not enough relevant test data to calculate Precision@10 and Recall@10.")

print("\n================================")
print("     EVALUATION COMPLETED!")
print("================================")
