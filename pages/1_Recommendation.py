
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.recommendation import recommend_movies
from src.collaborative_filtering import recommend_for_user


st.title("🎯 Movie Recommendations")

st.write(
    "Discover movies using intelligent machine learning recommendation algorithms."
)

# Load movies
@st.cache_data
def load_movies():
    path = BASE_DIR / "data" / "processed" / "movies_clean.csv"
    return pd.read_csv(path)


try:
    movies = load_movies()
except Exception as error:
    st.error(f"Unable to load movies: {error}")
    st.stop()


# Recommendation method
method = st.selectbox(
    "Choose Recommendation Method",
    [
        "Content-Based Filtering",
        "Collaborative Filtering"
    ]
)

st.divider()


# =====================================
# CONTENT BASED FILTERING
# =====================================

if method == "Content-Based Filtering":

    st.subheader("🎬 Find Similar Movies")

    movie_list = sorted(
        movies["title"].dropna().unique().tolist()
    )

    selected_movie = st.selectbox(
        "Select a movie",
        movie_list
    )

    number = st.slider(
        "Number of recommendations",
        5,
        20,
        10
    )

    if st.button("Recommend Similar Movies"):

        try:
            results = recommend_movies(
                selected_movie,
                number
            )

            st.subheader("Recommended Movies")

            if results is None or len(results) == 0:
                st.info("No recommendations found.")

            elif isinstance(results, pd.DataFrame):
                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True
                )

            else:
                for movie in results:
                    st.write("🎥", movie)

        except Exception as error:
            st.error(f"Recommendation failed: {error}")


# =====================================
# COLLABORATIVE FILTERING
# =====================================

elif method == "Collaborative Filtering":

    st.subheader("👥 User-Based Recommendations")

    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        value=1,
        step=1
    )

    number = st.slider(
        "Number of recommendations",
        5,
        20,
        10,
        key="collaborative_number"
    )

    if st.button("Recommend For User"):

        try:
            results = recommend_for_user(
                int(user_id),
                number
            )

            st.subheader("Recommended Movies")

            if results is None or len(results) == 0:
                st.info(
                    "No recommendations found for this user."
                )

            elif isinstance(results, pd.DataFrame):
                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True
                )

            else:
                for movie in results:
                    st.write("🎥", movie)

        except Exception as error:
            st.error(f"Recommendation failed: {error}")


st.divider()

st.caption(
    "MovieMind | Intelligent Movie Recommendation System"
)
