
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("\n================================")
print("       MOVIEMIND PROJECT")
print("       RECOMMENDATION ENGINE")
print("================================\n")

# Load cleaned movie dataset
movies = pd.read_csv("data/processed/movies_clean.csv")

# Handle missing genres
movies["genres"] = movies["genres"].fillna("")

# Convert genre text into TF-IDF vectors
tfidf = TfidfVectorizer(
    token_pattern=r"(?u)\b[\w-]+\b"
)

tfidf_matrix = tfidf.fit_transform(
    movies["genres"].str.replace("|", " ", regex=False)
)

print("TF-IDF MATRIX CREATED!")
print("Matrix Shape:", tfidf_matrix.shape)

# Movie recommendation function
def recommend_movies(movie_title, number_of_recommendations=10):

    # Find movie by exact title (case-insensitive)
    matches = movies[
        movies["title"].str.lower() == movie_title.strip().lower()
    ]

    if matches.empty:
        print("\nMovie not found!")
        print("Please enter an exact movie title.")
        return

    movie_index = matches.index[0]

    # Calculate similarity with selected movie
    similarity_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Sort movies by similarity
    similar_indices = similarity_scores.argsort()[::-1]

    # Exclude selected movie itself
    similar_indices = [
        i for i in similar_indices if i != movie_index
    ]

    # Select top recommendations
    top_indices = similar_indices[:number_of_recommendations]

    recommendations = movies.iloc[top_indices][
        ["title", "genres"]
    ].copy()

    recommendations["similarity_score"] = [
        round(float(similarity_scores[i]), 3)
        for i in top_indices
    ]

    print("\nMovies similar to:", movie_title)
    print("\n", recommendations.to_string(index=False))

    return recommendations


print("RECOMMENDATION ENGINE READY!")

# Test the recommendation system
recommend_movies("Toy Story (1995)", 10)
