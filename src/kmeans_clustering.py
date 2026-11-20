
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Load datasets
movies = pd.read_csv("data/processed/movies_clean.csv")
ratings = pd.read_csv("data/processed/ratings_clean.csv")

print("Movies:", movies.shape)
print("Ratings:", ratings.shape)


# Calculate movie-level rating features
movie_stats = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()


# Merge movie information
data = movies.merge(
    movie_stats,
    on="movieId",
    how="inner"
)

data = data.dropna(
    subset=["average_rating", "rating_count"]
)


# Prepare features
features = data[
    ["average_rating", "rating_count"]
].copy()

# Log transformation reduces rating-count skew
features["rating_count"] = np.log1p(
    features["rating_count"]
)


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)


# Apply K-Means
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

data["cluster"] = kmeans.fit_predict(X_scaled)


# Evaluation
score = silhouette_score(X_scaled, data["cluster"])

print("\n========== K-MEANS RESULTS ==========")
print("Number of clusters:", data["cluster"].nunique())
print("Silhouette Score:", round(score, 4))

print("\nMovies in each cluster:")
print(data["cluster"].value_counts().sort_index())


# Display sample movies from each cluster
print("\nSample Movies:")
for cluster_id in sorted(data["cluster"].unique()):
    print(f"\nCluster {cluster_id}:")
    print(
        data[data["cluster"] == cluster_id][
            ["title", "average_rating", "rating_count"]
        ].head(5).to_string(index=False)
    )


# Save results
os.makedirs("data/processed", exist_ok=True)

data.to_csv(
    "data/processed/kmeans_movie_clusters.csv",
    index=False
)


# Visualization
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=data,
    x="average_rating",
    y="rating_count",
    hue="cluster",
    palette="viridis",
    alpha=0.7
)

plt.yscale("log")
plt.title("MovieMind - K-Means Movie Clusters")
plt.xlabel("Average Rating")
plt.ylabel("Rating Count (log scale)")

plt.tight_layout()

plt.savefig(
    "data/processed/kmeans_clusters.png"
)

plt.close()

print("\nCluster CSV and graph saved successfully!")
print("\nK-MEANS CLUSTERING COMPLETED!")
