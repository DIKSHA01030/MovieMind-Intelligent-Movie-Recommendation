
import streamlit as st
import pandas as pd
import os
import random

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="MovieMind | Home",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #101014;
    color: white;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #F5C518;
}
.subtitle {
    color: #B8B8C2;
    font-size: 17px;
}
.movie-card {
    background: #202027;
    padding: 15px;
    border-radius: 12px;
    min-height: 135px;
    margin-bottom: 10px;
}
.movie-title {
    font-size: 17px;
    font-weight: bold;
    color: white;
}
.movie-info {
    font-size: 13px;
    color: #B8B8C2;
}
.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 25px;
}
div.stButton > button {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MOVIES_PATH = os.path.join(
    BASE_DIR, "data", "processed", "movies_clean.csv"
)

RATINGS_PATH = os.path.join(
    BASE_DIR, "data", "processed", "ratings_clean.csv"
)

@st.cache_data
def load_data():
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    stats = ratings.groupby("movieId").agg(
        average_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    movies = movies.merge(stats, on="movieId", how="left")

    movies["average_rating"] = movies["average_rating"].fillna(0)
    movies["rating_count"] = movies["rating_count"].fillna(0)

    return movies, ratings


try:
    movies, ratings = load_data()
except FileNotFoundError:
    st.error("Dataset files not found. Please check data/processed folder.")
    st.stop()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎬 MovieMind")
    st.caption("Intelligent Movie Discovery")

    st.divider()

    st.markdown("### Navigation")
    st.page_link("app.py", label="Main Dashboard", icon="🏠")
    st.page_link("pages/1_Home.py", label="Home", icon="🎬")
    st.page_link("pages/2_Movie_Search.py", label="Movie Search", icon="🔍")
    st.page_link("pages/3_Admin_Dashboard.py", label="Admin Dashboard", icon="📊")
    st.page_link("pages/4_SQL_Analytics.py", label="SQL Analytics", icon="🗄️")

    st.divider()
    st.caption("MovieMind 2.0 | Database + ML")


# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🎬 Welcome to MovieMind</div>',
            unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Discover movies. Explore genres. Find your next favourite.</div>',
    unsafe_allow_html=True
)

st.write("")


# ---------------- SEARCH BAR ----------------
st.markdown("### 🔍 Find Your Movie")

search = st.text_input(
    "Search",
    placeholder="Enter movie name...",
    label_visibility="collapsed"
)

if search:
    results = movies[
        movies["title"].astype(str).str.contains(
            search, case=False, na=False
        )
    ]

    st.markdown("#### Search Results")

    if len(results) > 0:
        for _, movie in results.head(10).iterrows():
            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-title">{movie['title']}</div>
                    <div class="movie-info">
                        Genre: {movie.get('genres', 'N/A')}<br>
                        ⭐ Rating: {movie['average_rating']:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No movies found. Try another title.")


# ---------------- STATISTICS ----------------
st.markdown('<div class="section-title">📊 Movie Collection</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Total Movies", f"{len(movies):,}")

with c2:
    st.metric("Total Ratings", f"{len(ratings):,}")

with c3:
    st.metric(
        "Average Rating",
        f"{ratings['rating'].mean():.2f} ⭐"
    )


# ---------------- MOVIE CARD FUNCTION ----------------
def show_movies(data):
    if data.empty:
        st.info("No movies available.")
        return

    cols = st.columns(4)

    for i, (_, movie) in enumerate(data.iterrows()):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-title">{movie['title']}</div>
                    <br>
                    <div class="movie-info">
                        🎭 {movie.get('genres', 'N/A')}<br><br>
                        ⭐ {movie['average_rating']:.2f}<br>
                        👥 {int(movie['rating_count'])} ratings
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ---------------- TRENDING MOVIES ----------------
st.markdown('<div class="section-title">🔥 Popular Movies</div>',
            unsafe_allow_html=True)

popular = movies.sort_values(
    by="rating_count",
    ascending=False
).head(8)

show_movies(popular)


# ---------------- TOP RATED MOVIES ----------------
st.markdown('<div class="section-title">⭐ Top Rated Movies</div>',
            unsafe_allow_html=True)

top_rated = movies[movies["rating_count"] >= 20].sort_values(
    by="average_rating",
    ascending=False
).head(8)

show_movies(top_rated)


# ---------------- GENRE SUGGESTIONS ----------------
st.markdown('<div class="section-title">🎭 Explore Genres</div>',
            unsafe_allow_html=True)

all_genres = sorted({
    genre
    for genres in movies["genres"].dropna()
    for genre in str(genres).split("|")
    if genre != "(no genres listed)"
})

selected_genre = st.selectbox(
    "Choose your favourite genre",
    ["Select Genre"] + all_genres
)

if selected_genre != "Select Genre":
    genre_movies = movies[
        movies["genres"].fillna("").str.contains(
            selected_genre, case=False, regex=False
        )
    ].sort_values("average_rating", ascending=False).head(8)

    st.markdown(f"#### 🎬 {selected_genre} Movies")
    show_movies(genre_movies)


# ---------------- SURPRISE ME ----------------
st.markdown('<div class="section-title">🎲 Surprise Me!</div>',
            unsafe_allow_html=True)

if st.button("Discover a Random Movie"):
    random_movie = movies.sample(1).iloc[0]

    st.success("Your random movie recommendation!")

    st.subheader(f"🎬 {random_movie['title']}")
    st.write(f"Genre: {random_movie['genres']}")
    st.write(f"Rating: ⭐ {random_movie['average_rating']:.2f}")


# ---------------- FOOTER ----------------
st.divider()

st.markdown(
    "<center style='color:#888'>"
    "MovieMind 2.0 | Intelligent Movie Discovery System"
    "</center>",
    unsafe_allow_html=True
)
