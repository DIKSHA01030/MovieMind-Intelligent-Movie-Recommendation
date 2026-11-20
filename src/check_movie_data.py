
import pandas as pd

movies = pd.read_csv("data/processed/movies_clean.csv")
ratings = pd.read_csv("data/processed/ratings_clean.csv")

print("MOVIES SHAPE:", movies.shape)
print("\nMOVIE COLUMNS:", movies.columns.tolist())

print("\nFIRST 5 MOVIES:")
print(movies.head())

print("\nGENRE VALUES:")
print(movies["genres"].head(10).tolist())

print("\nYEAR VALUES:")
if "year" in movies.columns:
    print(movies["year"].describe())
else:
    print("Year column is missing")

print("\nRATINGS SHAPE:", ratings.shape)
print("\nRATING COLUMNS:", ratings.columns.tolist())
print(ratings.head())
