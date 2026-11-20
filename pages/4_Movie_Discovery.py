from src.recommendation import recommend_movies
import streamlit as st
import pandas as pd
import random
import re
from pathlib import Path


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Movie Discovery | MovieMind",
    page_icon="🎬",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

.movie-card {
    background-color: #1e2530;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #343b48;
    min-height: 190px;
    margin-bottom: 15px;
}

.movie-title {
    font-size: 18px;
    font-weight: bold;
    color: #ffffff;
}

.movie-info {
    color: #c4c9d4;
    font-size: 14px;
    margin-top: 8px;
}

.discovery-heading {
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

MOVIES_PATH = BASE_DIR / "data" / "processed" / "movies_clean.csv"
RATINGS_PATH = BASE_DIR / "data" / "processed" / "ratings_clean.csv"


# ==========================================
# LOAD AND PREPARE DATA
# ==========================================

@st.cache_data
def load_movie_data():

    if not MOVIES_PATH.exists():
        raise FileNotFoundError(
            f"Movies file not found: {MOVIES_PATH}"
        )

    if not RATINGS_PATH.exists():
        raise FileNotFoundError(
            f"Ratings file not found: {RATINGS_PATH}"
        )

    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    # Check required columns
    if "movieId" not in movies.columns or "title" not in movies.columns:
        raise ValueError("movies_clean.csv must contain movieId and title columns.")

    if "movieId" not in ratings.columns or "rating" not in ratings.columns:
        raise ValueError("ratings_clean.csv must contain movieId and rating columns.")

    movies = movies.drop_duplicates(subset="movieId").copy()

    # Clean genres
    if "genres" not in movies.columns:
        movies["genres"] = "Unknown"

    movies["genres"] = (
        movies["genres"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    # Extract year from title if year column is missing
    if "year" not in movies.columns:
        movies["year"] = movies["title"].astype(str).str.extract(r"\((\d{4})\)")

    movies["year"] = pd.to_numeric(
        movies["year"], errors="coerce"
    )

    # Calculate movie rating statistics
    rating_stats = (
        ratings.groupby("movieId")["rating"]
        .agg(
            average_rating="mean",
            rating_count="count"
        )
        .reset_index()
    )

    # Merge ratings with movies
    movies = movies.merge(
        rating_stats,
        on="movieId",
        how="left"
    )

    movies["average_rating"] = movies["average_rating"].fillna(0)
    movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)

    return movies


# ==========================================
# GENRE HELPER
# ==========================================

def get_genres(genre_string):

    if pd.isna(genre_string):
        return []

    genres = str(genre_string).split("|")

    return [
        genre.strip()
        for genre in genres
        if genre.strip() and genre.strip() != "(no genres listed)"
    ]


# ==========================================
# LOAD DATA
# ==========================================

st.title("🎬 Movie Discovery")

st.markdown(
    "Explore thousands of movies and discover your next favourite film."
)

try:
    movies_df = load_movie_data()

except Exception as error:
    st.error(f"Unable to load movie data: {error}")
    st.stop()


if movies_df.empty:
    st.warning("No movies are available in the dataset.")
    st.stop()


# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("🔍 Movie Filters")

# Search
search_query = st.sidebar.text_input(
    "Search Movie",
    placeholder="Enter movie name..."
)

# Genre options
all_genres = sorted({
    genre
    for genre_string in movies_df["genres"]
    for genre in get_genres(genre_string)
})

selected_genres = st.sidebar.multiselect(
    "Select Genres",
    options=all_genres,
    default=[]
)

# Year filter
valid_years = movies_df["year"].dropna()

if not valid_years.empty:

    min_year = int(valid_years.min())
    max_year = int(valid_years.max())

    year_range = st.sidebar.slider(
        "Release Year",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

else:
    year_range = None


# Rating filter
minimum_rating = st.sidebar.slider(
    "Minimum Rating",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.5
)


# Sorting
sort_option = st.sidebar.selectbox(
    "Sort Movies By",
    [
        "Popularity",
        "Highest Rating",
        "Title A-Z",
        "Newest First"
    ]
)


# ==========================================
# APPLY FILTERS
# ==========================================

filtered_movies = movies_df.copy()

# Search filter
if search_query.strip():

    filtered_movies = filtered_movies[
        filtered_movies["title"].str.contains(
            search_query.strip(),
            case=False,
            na=False,
            regex=False
        )
    ]


# Genre filter
if selected_genres:

    def matches_genre(genre_string):
        movie_genres = get_genres(genre_string)

        # Match any selected genre
        return any(
            genre in movie_genres
            for genre in selected_genres
        )

    filtered_movies = filtered_movies[
        filtered_movies["genres"].apply(matches_genre)
    ]


# Year filter
if year_range is not None:

    filtered_movies = filtered_movies[
        filtered_movies["year"].between(
            year_range[0],
            year_range[1]
        )
    ]


# Rating filter
filtered_movies = filtered_movies[
    filtered_movies["average_rating"] >= minimum_rating
]


# ==========================================
# SORT RESULTS
# ==========================================

if sort_option == "Popularity":

    filtered_movies = filtered_movies.sort_values(
        by="rating_count",
        ascending=False
    )

elif sort_option == "Highest Rating":

    filtered_movies = filtered_movies.sort_values(
        by=["average_rating", "rating_count"],
        ascending=[False, False]
    )

elif sort_option == "Title A-Z":

    filtered_movies = filtered_movies.sort_values(
        by="title",
        ascending=True
    )

elif sort_option == "Newest First":

    filtered_movies = filtered_movies.sort_values(
        by="year",
        ascending=False,
        na_position="last"
    )


# ==========================================
# MOVIE STATISTICS
# ==========================================

st.divider()

st.subheader("📊 Movie Statistics")

col1, col2, col3 = st.columns(3)

total_movies = len(filtered_movies)

if total_movies > 0:

    average_rating = filtered_movies[
        filtered_movies["average_rating"] > 0
    ]["average_rating"].mean()

    total_ratings = int(filtered_movies["rating_count"].sum())

else:
    average_rating = 0
    total_ratings = 0


with col1:
    st.metric(
        "Total Movies Found",
        f"{total_movies:,}"
    )

with col2:
    st.metric(
        "Average Rating",
        f"{average_rating:.2f}/5" if pd.notna(average_rating) and average_rating > 0 else "N/A"
    )

with col3:
    st.metric(
        "Total Ratings",
        f"{total_ratings:,}"
    )


# ==========================================
# EXPLORE MOVIES
# ==========================================

st.divider()

st.subheader("🎬 Explore Movies")

if filtered_movies.empty:

    st.warning(
        "No movies found. Please change your search or filters."
    )

else:

    st.success(
        f"{len(filtered_movies):,} movies found!"
    )

    # Pagination
    movies_per_page = 12

    total_pages = (
        len(filtered_movies) + movies_per_page - 1
    ) // movies_per_page

    if "discovery_page" not in st.session_state:
        st.session_state.discovery_page = 1

    current_page = st.number_input(
        "Page",
        min_value=1,
        max_value=max(1, total_pages),
        value=min(st.session_state.discovery_page, max(1, total_pages)),
        step=1
    )

    st.session_state.discovery_page = current_page

    start_index = (current_page - 1) * movies_per_page
    end_index = start_index + movies_per_page

    page_movies = filtered_movies.iloc[
        start_index:end_index
    ]

    # Display movie cards
    movie_rows = page_movies.to_dict("records")

    for i in range(0, len(movie_rows), 3):

        columns = st.columns(3)

        for j, movie in enumerate(movie_rows[i:i + 3]):

            with columns[j]:

                title = movie["title"]

                genres = ", ".join(
                    get_genres(movie["genres"])
                ) or "Unknown"

                year = (
                    str(int(movie["year"]))
                    if pd.notna(movie["year"])
                    else "Unknown"
                )

                rating = movie["average_rating"]
                count = movie["rating_count"]

                with st.container(border=True):

                    st.markdown(
                        f"**🎥 {title}**"
                    )

                    st.caption(f"📅 Year: {year}")

                    st.write(f"🎭 {genres}")

                    if rating > 0:
                        st.write(f"⭐ Rating: {rating:.2f}/5")
                    else:
                        st.write("⭐ No ratings available")

                    st.write(f"👥 Total Ratings: {count:,}")


    st.caption(
        f"Showing {start_index + 1}–{min(end_index, len(filtered_movies))} "
        f"of {len(filtered_movies)} movies"
    )


# ==========================================
# SURPRISE ME
# ==========================================

st.divider()

st.subheader("🎲 Surprise Me!")

st.write(
    "Can't decide what to watch? Let MovieMind select a random movie for you!"
)

if st.button("🎲 Surprise Me!", use_container_width=True):

    if not filtered_movies.empty:

        random_movie = filtered_movies.sample(
            n=1
        ).iloc[0]

        st.session_state["surprise_movie"] = random_movie.to_dict()

    else:

        st.session_state.pop("surprise_movie", None)

        st.warning(
            "No movies available with the current filters."
        )


if "surprise_movie" in st.session_state:

    surprise = st.session_state["surprise_movie"]

    st.success("✨ Your Movie Discovery!")

    st.markdown(f"### 🎬 {surprise['title']}")

    surprise_genres = ", ".join(
        get_genres(surprise["genres"])
    ) or "Unknown"

    surprise_year = (
        str(int(surprise["year"]))
        if pd.notna(surprise["year"])
        else "Unknown"
    )

    st.write(f"**Genres:** {surprise_genres}")
    st.write(f"**Release Year:** {surprise_year}")

    if surprise["average_rating"] > 0:
        st.write(
            f"**Rating:** ⭐ {surprise['average_rating']:.2f}/5"
        )

    st.write(
        f"**Total Ratings:** {surprise['rating_count']:,}"
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        🎬 <b>MovieMind</b> | Intelligent Movie Recommendation System
        <br>
        Powered by Python, Pandas, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
