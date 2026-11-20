
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("\n================================")
print("       MOVIEMIND PROJECT")
print("      HYBRID RECOMMENDER")
print("================================\n")

# Load datasets
movies = pd.read_csv("data/processed/movies_clean.csv")
ratings = pd.read_csv("data/processed/ratings_clean.csv")

movies["genres"] = movies["genres"].fillna("")

# --------------------------------
# PART 1: CONTENT-BASED FILTERING
# --------------------------------

tfidf = TfidfVectorizer(
    token_pattern=r"(?u)\b[\w-]+\b"
)

tfidf_matrix = tfidf.fit_transform(
    movies["genres"].str.replace("|", " ", regex=False)
)

content_similarity = cosine_similarity(tfidf_matrix)

movie_id_to_index = pd.Series(
    movies.index,
    index=movies["movieId"]
)

# --------------------------------
# PART 2: COLLABORATIVE FILTERING
# --------------------------------

rating_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

filled_matrix = rating_matrix.fillna(0)

user_similarity = cosine_similarity(filled_matrix)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=rating_matrix.index,
    columns=rating_matrix.index
)


# --------------------------------
# PART 3: HYBRID RECOMMENDATION
# --------------------------------

def hybrid_recommend(user_id, movie_title, top_n=10, alpha=0.5):

    # Check user
    if user_id not in rating_matrix.index:
        print("User ID not found!")
        return

    # Find selected movie
    matches = movies[
        movies["title"].str.lower() == movie_title.lower()
    ]

    if matches.empty:
        print("Movie not found!")
        return

    movie_id = matches.iloc[0]["movieId"]

    movie_index = movie_id_to_index[movie_id]

    # Content similarity scores
    content_scores = content_similarity[movie_index]

    # Similar users
    similar_users = (
        user_similarity_df.loc[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
        .head(10)
    )

    neighbor_ratings = rating_matrix.loc[similar_users.index]

    weighted_sum = neighbor_ratings.mul(
        similar_users, axis=0
    ).sum(axis=0)

    weight_sum = (
        neighbor_ratings.notna()
        .mul(similar_users, axis=0)
        .sum(axis=0)
    )

    collaborative_scores = weighted_sum.div(
        weight_sum.replace(0, np.nan)
    )

    # Normalize predicted ratings to 0-1
    collaborative_scores = (
        (collaborative_scores - 0.5) / 4.5
    ).clip(0, 1)

    # Create final result
    results = movies[["movieId", "title", "genres"]].copy()

    results["content_score"] = results["movieId"].map(
        lambda mid: content_scores[
            movie_id_to_index[mid]
        ]
    )

    results["collaborative_score"] = (
        results["movieId"].map(collaborative_scores)
    )

    # Combine both scores
    results["hybrid_score"] = (
        alpha * results["content_score"]
        + (1 - alpha) * results["collaborative_score"]
    )

    # Exclude selected movie and already-rated movies
    already_rated = rating_matrix.loc[user_id].dropna().index

    results = results[
        (results["movieId"] != movie_id)
        & (~results["movieId"].isin(already_rated))
    ]

    # Exclude movies without collaborative predictions
    results = results.dropna(subset=["collaborative_score"])

    results = results.sort_values(
        "hybrid_score",
        ascending=False
    ).head(top_n)

    print("\nHybrid Recommendations")
    print("User ID:", user_id)
    print("Based on:", movie_title)

    print(
        results[
            [
                "title",
                "genres",
                "content_score",
                "collaborative_score",
                "hybrid_score"
            ]
        ].round(3).to_string(index=False)
    )

    return results


print("HYBRID ENGINE READY!")

# Test
hybrid_recommend(
    user_id=1,
    movie_title="Toy Story (1995)",
    top_n=10,
    alpha=0.5
)
