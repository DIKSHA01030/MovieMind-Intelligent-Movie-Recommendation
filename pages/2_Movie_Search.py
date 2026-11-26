
import streamlit as st
import pandas as pd
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Movie Search | MovieMind",
    page_icon="🔍",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #101014;
    color: white;
}
.page-title {
    font-size: 38px;
    font-weight: 800;
    color: #F5C518;
}
.movie-card {
    background: #202027;
    padding: 17px;
    border-radius: 12px;
    min-height: 170px;
    margin-bottom: 15px;
    border: 1px solid #33333D;
}
.movie-title {
    font-size: 17px;
    font-weight: bold;
    color: white;
}
.movie-info {
    color: #B8B8C2;
    font-size: 13px;
    line-height: 1.8;
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

    return movies

try:
    movies = load_data()
except FileNotFoundError:
    st.error("Movie dataset not found. Check data/processed folder.")
    st.stop()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎬 MovieMind")
    st.caption("Movie Discovery Platform")

    st.divider()

    st.page_link("app.py", label="Main Dashboard", icon="🏠")
    st.page_link("pages/1_Home.py", label="Home", icon="🎬")
    st.page_link("pages/2_Movie_Search.py", label="Movie Search", icon="🔍")
    st.page_link("pages/3_Admin_Dashboard.py", label="Admin Dashboard", icon="📊")
    st.page_link("pages/4_SQL_Analytics.py", label="SQL Analytics", icon="🗄️")


# ---------------- HEADER ----------------
st.markdown(
    '<div class="page-title">🔍 Explore Movies</div>',
    unsafe_allow_html=True
)

st.write("Search and filter movies from the MovieMind collection.")

st.divider()


# ---------------- SEARCH AND FILTERS ----------------
st.subheader("🎯 Search Filters")

col1, col2 = st.columns([2, 1])

with col1:
    search_query = st.text_input(
        "Movie Title",
        placeholder="Search by movie name..."
    )

with col2:
    genre_options = sorted({
        genre
        for genres in movies["genres"].dropna()
        for genre in str(genres).split("|")
        if genre != "(no genres listed)"
    })

    selected_genre = st.selectbox(
        "Genre",
        ["All Genres"] + genre_options
    )

col3, col4, col5 = st.columns(3)

with col3:
    years = pd.to_numeric(movies["year"], errors="coerce").dropna()

    if not years.empty:
        min_year = int(years.min())
        max_year = int(years.max())

        year_range = st.slider(
            "Release Year",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
    else:
        year_range = None

with col4:
    min_rating = st.slider(
        "Minimum Rating",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.5
    )

with col5:
    sort_option = st.selectbox(
        "Sort By",
        [
            "Popularity",
            "Highest Rating",
            "Title A-Z",
            "Newest First"
        ]
    )


# ---------------- APPLY FILTERS ----------------
filtered = movies.copy()

if search_query:
    filtered = filtered[
        filtered["title"].astype(str).str.contains(
            search_query,
            case=False,
            na=False
        )
    ]

if selected_genre != "All Genres":
    filtered = filtered[
        filtered["genres"].fillna("").str.contains(
            selected_genre,
            case=False,
            regex=False
        )
    ]

if year_range is not None:
    filtered = filtered[
        pd.to_numeric(filtered["year"], errors="coerce").between(
            year_range[0], year_range[1]
        )
    ]

filtered = filtered[
    filtered["average_rating"] >= min_rating
]


# ---------------- SORTING ----------------
if sort_option == "Popularity":
    filtered = filtered.sort_values(
        "rating_count", ascending=False
    )

elif sort_option == "Highest Rating":
    filtered = filtered.sort_values(
        "average_rating", ascending=False
    )

elif sort_option == "Title A-Z":
    filtered = filtered.sort_values(
        "title", ascending=True
    )

elif sort_option == "Newest First":
    filtered = filtered.sort_values(
        "year", ascending=False, na_position="last"
    )


# ---------------- RESULTS ----------------
st.divider()

st.subheader("🎬 Movie Search Results")

a, b = st.columns(2)

with a:
    st.metric("Movies Found", len(filtered))

with b:
    avg = filtered["average_rating"].mean() if not filtered.empty else 0
    st.metric("Average Rating", f"{avg:.2f} ⭐")


if filtered.empty:
    st.warning("No movies found. Try changing your filters.")

else:
    st.write("")

    # Display results in 4 columns
    cols = st.columns(4)

    for i, (_, movie) in enumerate(filtered.iterrows()):
        with cols[i % 4]:

            year = movie.get("year", "N/A")

            if pd.isna(year):
                year = "N/A"
            else:
                year = int(year)

            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-title">
                        🎬 {movie['title']}
                    </div>
                    <br>
                    <div class="movie-info">
                        📅 Year: {year}<br>
                        🎭 Genre: {movie['genres']}<br>
                        ⭐ Rating: {movie['average_rating']:.2f}/5<br>
                        👥 Ratings: {int(movie['rating_count'])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ---------------- DOWNLOAD RESULTS ----------------
st.divider()

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Search Results (CSV)",
    data=csv_data,
    file_name="moviemind_search_results.csv",
    mime="text/csv"
)


# ---------------- FOOTER ----------------
st.divider()

st.caption("MovieMind 2.0 | Movie Search & Discovery")
