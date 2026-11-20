
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("data/processed", exist_ok=True)

print("\n================================")
print("       MOVIEMIND PROJECT")
print("       DATA PREPROCESSING")
print("================================\n")

# Load datasets
movies = pd.read_csv("data/raw/ml-latest-small/movies.csv")
ratings = pd.read_csv("data/raw/ml-latest-small/ratings.csv")
tags = pd.read_csv("data/raw/ml-latest-small/tags.csv")

print("DATASETS LOADED SUCCESSFULLY!")

# 1. Check missing values
print("\n--- MISSING VALUES ---")
print("Movies:\n", movies.isnull().sum())
print("Ratings:\n", ratings.isnull().sum())
print("Tags:\n", tags.isnull().sum())

# 2. Check duplicates
print("\n--- DUPLICATE RECORDS ---")
print("Movies:", movies.duplicated().sum())
print("Ratings:", ratings.duplicated().sum())
print("Tags:", tags.duplicated().sum())

# 3. Remove duplicate records
movies = movies.drop_duplicates()
ratings = ratings.drop_duplicates()
tags = tags.drop_duplicates()

# 4. Handle missing values
movies = movies.dropna(subset=["movieId", "title"])
ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
tags = tags.dropna(subset=["userId", "movieId", "tag"])

# 5. Validate rating range
ratings = ratings[
    ratings["rating"].between(0.5, 5.0)
]

# 6. Extract movie release year
movies["year"] = (
    movies["title"]
    .str.extract(r"\((\d{4})\)$")
)

# 7. Create genre list
movies["genres_list"] = movies["genres"].apply(
    lambda x: x.split("|") if pd.notna(x) and x != "(no genres listed)" else []
)

# 8. Display cleaned dataset information
print("\n--- CLEANED DATASET INFORMATION ---")
print("Movies:", movies.shape)
print("Ratings:", ratings.shape)
print("Tags:", tags.shape)

print("\n--- RATING STATISTICS ---")
print(ratings["rating"].describe())

print("\nAverage Rating:", ratings["rating"].mean())

# 9. Find most rated movies
movie_counts = (
    ratings.groupby("movieId")
    .size()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name="rating_count")
)

top_movies = movie_counts.merge(movies, on="movieId")

print("\n--- TOP 10 MOST RATED MOVIES ---")
print(top_movies[["title", "rating_count"]])

# 10. Plot rating distribution
plt.figure(figsize=(8, 5))
plt.hist(ratings["rating"], bins=10, edgecolor="black")
plt.title("Movie Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")
plt.tight_layout()
plt.savefig("data/processed/rating_distribution.png")
plt.show()

# 11. Save cleaned datasets
movies.to_csv("data/processed/movies_clean.csv", index=False)
ratings.to_csv("data/processed/ratings_clean.csv", index=False)
tags.to_csv("data/processed/tags_clean.csv", index=False)

print("\n================================")
print("PREPROCESSING COMPLETED!")
print("CLEANED DATASETS SAVED!")
print("================================")
