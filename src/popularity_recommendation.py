
import pandas as pd
import os


# Load datasets
movies = pd.read_csv("data/processed/movies_clean.csv")
ratings = pd.read_csv("data/processed/ratings_clean.csv")

print("Movies:", movies.shape)
print("Ratings:", ratings.shape)


# Calculate movie statistics
movie_stats = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()


# Calculate overall average rating
C = ratings["rating"].mean()

# Minimum votes required
m = ratings["movieId"].value_counts().quantile(0.70)

print("\nOverall Average Rating:", round(C, 3))
print("Minimum Rating Threshold:", round(m, 2))


# Merge movie information
data = movies.merge(
    movie_stats,
    on="movieId",
    how="inner"
)


# Calculate weighted rating
data["weighted_rating"] = (
    (data["rating_count"] / (data["rating_count"] + m))
    * data["average_rating"]
    +
    (m / (data["rating_count"] + m))
    * C
)


# Sort movies by weighted rating
popular_movies = data.sort_values(
    by="weighted_rating",
    ascending=False
)


# Display top 20 movies
print("\n========== TOP 20 POPULAR MOVIES ==========")

print(
    popular_movies[
        ["title", "average_rating", "rating_count", "weighted_rating"]
    ].head(20).to_string(index=False)
)


# Save results
os.makedirs("data/processed", exist_ok=True)

popular_movies.to_csv(
    "data/processed/popular_movies.csv",
    index=False
)


print("\nPopularity recommendations saved!")
print("\nPOPULARITY MODEL COMPLETED!")

