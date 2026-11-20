
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# Load datasets
movies = pd.read_csv("data/processed/movies_clean.csv")
tags = pd.read_csv("data/processed/tags_clean.csv")

print("Movies Dataset:", movies.shape)
print("Tags Dataset:", tags.shape)


# Merge tags with movie information
data = tags.merge(
    movies[["movieId", "genres"]],
    on="movieId",
    how="inner"
)

# Remove missing values
data = data.dropna(subset=["tag", "genres"])

# Remove rows without useful text or genre
data["tag"] = data["tag"].astype(str).str.strip()
data = data[data["tag"] != ""]
data = data[data["genres"] != "(no genres listed)"]

# Select one genre per tag row for a simple multiclass demo
data["genre"] = data["genres"].str.split("|").str[0]

print("\nPrepared Dataset:", data.shape)
print("\nGenre Distribution:")
print(data["genre"].value_counts().head(10))


# Features and target
X = data["tag"]
y = data["genre"]


# Remove classes having fewer than 2 examples
counts = y.value_counts()
valid_genres = counts[counts >= 2].index

data = data[data["genre"].isin(valid_genres)]

X = data["tag"]
y = data["genre"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)


# Predictions
predictions = model.predict(X_test_vectorized)


# Evaluation
accuracy = accuracy_score(y_test, predictions)

print("\n========== NAIVE BAYES RESULTS ==========")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))


# Confusion Matrix
labels = model.classes_

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

os.makedirs("data/processed", exist_ok=True)

plt.figure(figsize=(10, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Naive Bayes - Genre Classification")
plt.xlabel("Predicted Genre")
plt.ylabel("Actual Genre")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig("data/processed/naive_bayes_confusion_matrix.png")
plt.close()


print("\nConfusion matrix saved successfully!")
print("\nNAIVE BAYES MODEL COMPLETED!")
