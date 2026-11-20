
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("\n================================")
print("       MOVIEMIND PROJECT")
print("       LOGISTIC REGRESSION")
print("================================\n")

# 1. Load datasets
ratings = pd.read_csv("data/processed/ratings_clean.csv")

# 2. Create user statistics
user_stats = ratings.groupby("userId").agg(
    user_avg_rating=("rating", "mean"),
    user_rating_count=("rating", "count")
).reset_index()

# 3. Create movie statistics
movie_stats = ratings.groupby("movieId").agg(
    movie_avg_rating=("rating", "mean"),
    movie_rating_count=("rating", "count")
).reset_index()

# 4. Merge features
data = ratings.merge(user_stats, on="userId")
data = data.merge(movie_stats, on="movieId")

# 5. Convert ratings into Like/Dislike labels
data["liked"] = (data["rating"] >= 4.0).astype(int)

# 6. Select features
X = data[
    [
        "user_avg_rating",
        "user_rating_count",
        "movie_avg_rating",
        "movie_rating_count"
    ]
]

y = data["liked"]

# 7. Split dataset using stratification
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# 8. Create Logistic Regression pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 9. Train model
model.fit(X_train, y_train)

print("\nLOGISTIC REGRESSION MODEL TRAINED!")

# 10. Make predictions
predictions = model.predict(X_test)

# 11. Calculate evaluation metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, predictions))

# 12. Confusion Matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks([0, 1], ["Dislike", "Like"])
plt.yticks([0, 1], ["Dislike", "Like"])
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()

plt.savefig(
    "data/processed/logistic_confusion_matrix.png"
)

plt.show()

print("\n================================")
print("LOGISTIC REGRESSION COMPLETED!")
print("================================")
