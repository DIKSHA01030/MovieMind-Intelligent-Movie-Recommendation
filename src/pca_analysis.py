
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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

# Merge movie information
data = movies.merge(
    movie_stats,
    on="movieId",
    how="inner"
)


# Select numerical features
features = data[
    ["average_rating", "rating_count", "rating_std"]
].copy()

features["rating_count"] = np.log1p(
    features["rating_count"]
)


# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)


# Apply PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


# Add PCA components
data["PC1"] = X_pca[:, 0]
data["PC2"] = X_pca[:, 1]


# Explained variance
print("\n========== PCA RESULTS ==========")

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print(
    "Total Variance Preserved:",
    round(sum(pca.explained_variance_ratio_) * 100, 2),
    "%"
)


# Save transformed dataset
os.makedirs("data/processed", exist_ok=True)

data.to_csv(
    "data/processed/pca_movie_data.csv",
    index=False
)


# Visualize PCA
plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=data,
    x="PC1",
    y="PC2",
    alpha=0.6
)

plt.title("MovieMind - PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()

plt.savefig(
    "data/processed/pca_visualization.png"
)

plt.close()


print("\nPCA graph saved successfully!")
print("\nPCA ANALYSIS COMPLETED!")
