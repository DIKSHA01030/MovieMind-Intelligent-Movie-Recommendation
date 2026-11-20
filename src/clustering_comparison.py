
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Load datasets
movies = pd.read_csv("data/processed/movies_clean.csv")
ratings = pd.read_csv("data/processed/ratings_clean.csv")

# Movie-level features
movie_stats = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count"),
    rating_std=("rating", "std")
).reset_index()

movie_stats["rating_std"] = movie_stats["rating_std"].fillna(0)

data = movies.merge(movie_stats, on="movieId", how="inner")

features = data[
    ["average_rating", "rating_count", "rating_std"]
].copy()

features["rating_count"] = np.log1p(features["rating_count"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)


# Load previous clustering results
kmeans_data = pd.read_csv(
    "data/processed/kmeans_movie_clusters.csv"
)

em_data = pd.read_csv(
    "data/processed/em_movie_clusters.csv"
)

# Align labels using movieId
comparison = data[["movieId", "title"]].copy()

comparison = comparison.merge(
    kmeans_data[["movieId", "cluster"]],
    on="movieId",
    how="inner"
)

comparison = comparison.merge(
    em_data[["movieId", "EM_Cluster"]],
    on="movieId",
    how="inner"
)


# Calculate clustering metrics
kmeans_labels = comparison["cluster"].to_numpy()
em_labels = comparison["EM_Cluster"].to_numpy()

results = []

for name, labels in [
    ("K-Means", kmeans_labels),
    ("Gaussian Mixture Model (EM)", em_labels)
]:
    results.append({
        "Model": name,
        "Silhouette Score": silhouette_score(X_scaled, labels),
        "Davies-Bouldin Score": davies_bouldin_score(X_scaled, labels),
        "Calinski-Harabasz Score": calinski_harabasz_score(X_scaled, labels)
    })


results_df = pd.DataFrame(results)

print("\n========== CLUSTERING COMPARISON ==========")
print(results_df.round(4).to_string(index=False))


# Save comparison
os.makedirs("data/processed", exist_ok=True)

results_df.to_csv(
    "data/processed/clustering_comparison.csv",
    index=False
)


# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

comparison["PC1"] = X_pca[:, 0]
comparison["PC2"] = X_pca[:, 1]

# K-Means plot
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=comparison,
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="viridis",
    alpha=0.65
)

plt.title("K-Means Clusters - PCA Projection")
plt.tight_layout()

plt.savefig(
    "data/processed/kmeans_pca_comparison.png"
)
plt.close()


# EM plot
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=comparison,
    x="PC1",
    y="PC2",
    hue="EM_Cluster",
    palette="viridis",
    alpha=0.65
)

plt.title("EM Clusters - PCA Projection")
plt.tight_layout()

plt.savefig(
    "data/processed/em_pca_comparison.png"
)
plt.close()


print("\nComparison CSV and PCA graphs saved!")
print("\nCLUSTERING COMPARISON COMPLETED!")
