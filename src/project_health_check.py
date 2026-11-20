
from pathlib import Path
import pandas as pd
import sys

BASE_DIR = Path(__file__).resolve().parents[1]

print("\n====================================")
print("       MOVIEMIND HEALTH CHECK")
print("====================================")

# 1. Check important files
required_files = [
    "data/processed/movies_clean.csv",
    "data/processed/ratings_clean.csv",
    "data/processed/tags_clean.csv",
    "app.py",
    "src/recommendation.py",
    "src/collaborative_filtering.py",
    "src/hybrid_recommendation.py",
    "pages/1_Recommendations.py",
    "pages/2_Advanced_Recommendations.py",
    "pages/3_ML_Analytics.py",
    "pages/4_Movie_Discovery.py"
]

print("\n1. FILE CHECK")

missing_files = []

for file in required_files:
    path = BASE_DIR / file

    if path.exists():
        print(f"[PASS] {file}")
    else:
        print(f"[FAIL] {file}")
        missing_files.append(file)


# 2. Dataset validation
print("\n2. DATASET CHECK")

try:
    movies = pd.read_csv(
        BASE_DIR / "data/processed/movies_clean.csv"
    )

    ratings = pd.read_csv(
        BASE_DIR / "data/processed/ratings_clean.csv"
    )

    tags = pd.read_csv(
        BASE_DIR / "data/processed/tags_clean.csv"
    )

    print("Movies:", movies.shape)
    print("Ratings:", ratings.shape)
    print("Tags:", tags.shape)

    required_movie_columns = {"movieId", "title", "genres"}
    required_rating_columns = {"userId", "movieId", "rating"}

    if required_movie_columns.issubset(movies.columns):
        print("[PASS] Movie columns")
    else:
        print("[FAIL] Movie columns")

    if required_rating_columns.issubset(ratings.columns):
        print("[PASS] Rating columns")
    else:
        print("[FAIL] Rating columns")

    if not movies.empty and not ratings.empty:
        print("[PASS] Dataset is not empty")
    else:
        print("[FAIL] Empty dataset")

except Exception as error:
    print("[FAIL] Dataset loading:", error)


# 3. Recommendation module check
print("\n3. RECOMMENDATION CHECK")

try:
    sys.path.insert(0, str(BASE_DIR))

    from src.recommendation import recommend_movies

    print("[PASS] Recommendation module imported")

    result = recommend_movies("Toy Story (1995)", 5)

    if result is not None and len(result) > 0:
        print("[PASS] Recommendations generated")
        print(result)

    else:
        print("[WARNING] No recommendations returned")

except Exception as error:
    print("[FAIL] Recommendation test:", error)


# 4. Final summary
print("\n====================================")
print("          HEALTH CHECK SUMMARY")
print("====================================")

if not missing_files:
    print("All required project files are present.")
else:
    print("Missing files:", len(missing_files))

print("Health check completed.")
if not missing_files:
    print("All required project files are present.")
else:
    print("Missing files:", len(missing_files))
    print("\nMissing file names:")
    for file in missing_files:
        print("[FAIL]", file)