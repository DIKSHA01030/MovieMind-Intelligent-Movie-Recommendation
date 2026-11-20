
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


# Load datasets
movies = pd.read_csv("data/processed/movies_clean.csv")
ratings = pd.read_csv("data/processed/ratings_clean.csv")

print("Movies:", movies.shape)
print("Ratings:", ratings.shape)


# Calculate movie statistics
movie_stats = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count"),
    rating_std=("rating", "std")
).reset_index()

movie_stats["rating_std"] = (
    movie_stats["rating_std"].fillna(0)
)


# Merge datasets
data = movies.merge(
    movie_stats,
    on="movieId",
    how="inner"
)


# Prepare features
features = data[
    ["average_rating", "rating_count", "rating_std"]
].copy()

features["rating_count"] = np.log1p(
    features["rating_count"]
)


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)


# Apply Gaussian Mixture Model
gmm = GaussianMixture(
    n_components=5,
    covariance_type="full",
    random_state=42,
    n_init=3
)

gmm.fit(X_scaled)


# Predict cluster labels
data["EM_Cluster"] = gmm.predict(X_scaled)


# Calculate cluster probabilities
probabilities = gmm.predict_proba(X_scaled)

data["Cluster_Confidence"] = probabilities.max(axis=1)


# Evaluation
silhouette = silhouette_score(
    X_scaled,
    data["EM_Cluster"]
)

print("\n========== EM ALGORITHM RESULTS ==========")

print("Number of Clusters:", data["EM_Cluster"].nunique())

print("Silhouette Score:", round(silhouette, 4))

print(
    "Model Converged:",
    gmm.converged_
)

print(
    "Number of Iterations:",
    gmm.n_iter_
)

print("\nCluster Distribution:")
print(data["EM_Cluster"].value_counts().sort_index())


# Save results
os.makedirs("data/processed", exist_ok=True)

data.to_csv(
    "data/processed/em_movie_clusters.csv",
    index=False
)


# Visualization
plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=data,
    x="average_rating",
    y="rating_count",
    hue="EM_Cluster",
    palette="viridis",
    alpha=0.7
)

plt.yscale("log")
plt.title("MovieMind - EM Clustering")
plt.xlabel("Average Rating")
plt.ylabel("Rating Count (log scale)")

plt.tight_layout()

plt.savefig(
    "data/processed/em_clusters.png"
)

plt.close()


print("\nEM clustering graph saved!")
print("\nEM ALGORITHM COMPLETED!")
