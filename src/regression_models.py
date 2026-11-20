
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n================================")
print("       MOVIEMIND PROJECT")
print("       LINEAR REGRESSION")
print("================================\n")

# 1. Load datasets
ratings = pd.read_csv("data/processed/ratings_clean.csv")
movies = pd.read_csv("data/processed/movies_clean.csv")

# 2. Calculate average rating for every movie
movie_stats = ratings.groupby("movieId").agg(
    movie_avg_rating=("rating", "mean"),
    movie_rating_count=("rating", "count")
).reset_index()

# 3. Calculate average rating for every user
user_stats = ratings.groupby("userId").agg(
    user_avg_rating=("rating", "mean"),
    user_rating_count=("rating", "count")
).reset_index()

# 4. Merge features
data = ratings.merge(movie_stats, on="movieId")
data = data.merge(user_stats, on="userId")

# 5. Select input features and target
X = data[
    [
        "user_avg_rating",
        "user_rating_count",
        "movie_avg_rating",
        "movie_rating_count"
    ]
]

y = data["rating"]

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# 7. Train Linear Regression
model = LinearRegression()

model.fit(X_train, y_train)

print("\nLINEAR REGRESSION MODEL TRAINED!")

# 8. Predict ratings
predictions = model.predict(X_test)

predictions = np.clip(predictions, 0.5, 5.0)

# 9. Evaluate model
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")

print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

# 10. Display actual vs predicted
results = pd.DataFrame({
    "Actual Rating": y_test.values[:10],
    "Predicted Rating": predictions[:10]
})

print("\n========== SAMPLE PREDICTIONS ==========")
print(results.to_string(index=False))

# 11. Plot actual vs predicted ratings
plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    predictions,
    alpha=0.3
)

plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Movie Ratings")

plt.tight_layout()

plt.savefig(
    "data/processed/linear_regression_results.png"
)

plt.show()

print("\n================================")
print("LINEAR REGRESSION COMPLETED!")
print("================================")
