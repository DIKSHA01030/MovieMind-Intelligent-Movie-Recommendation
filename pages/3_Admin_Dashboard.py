
import streamlit as st
import pandas as pd
import os
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Admin Dashboard | MovieMind",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #101014;
    color: white;
}
.dashboard-title {
    font-size: 38px;
    font-weight: 800;
    color: #F5C518;
}
.metric-card {
    background-color: #202027;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #33333D;
}
.section-title {
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- DATA LOADING ----------------
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
    st.error("Dataset files not found. Check data/processed folder.")
    st.stop()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎬 MovieMind")
    st.caption("Administrator Panel")

    st.divider()

    st.page_link("app.py", label="Main Dashboard", icon="🏠")
    st.page_link("pages/1_Home.py", label="Home", icon="🎬")
    st.page_link("pages/2_Movie_Search.py", label="Movie Search", icon="🔍")
    st.page_link("pages/3_Admin_Dashboard.py", label="Admin Dashboard", icon="📊")
    st.page_link("pages/4_SQL_Analytics.py", label="SQL Analytics", icon="🗄️")

    st.divider()
    st.caption("MovieMind 2.0")


# ---------------- HEADER ----------------
st.markdown(
    '<div class="dashboard-title">📊 Admin Dashboard</div>',
    unsafe_allow_html=True
)

st.write("Monitor your movie collection and platform statistics.")

st.divider()


# ---------------- KPI METRICS ----------------
st.subheader("📌 Platform Overview")

total_movies = len(movies)
total_ratings = len(ratings)
total_users = ratings["userId"].nunique()
average_rating = ratings["rating"].mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric("🎬 Total Movies", f"{total_movies:,}")
c2.metric("⭐ Total Ratings", f"{total_ratings:,}")
c3.metric("👥 Total Users", f"{total_users:,}")
c4.metric("📈 Average Rating", f"{average_rating:.2f}/5")


# ---------------- MOVIE STATISTICS ----------------
st.markdown(
    '<div class="section-title">🎥 Movie Statistics</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    popular = movies.sort_values(
        "rating_count", ascending=False
    ).head(10)

    fig = px.bar(
        popular.sort_values("rating_count"),
        x="rating_count",
        y="title",
        orientation="h",
        title="Top 10 Popular Movies",
        labels={
            "rating_count": "Number of Ratings",
            "title": "Movie"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101014",
        plot_bgcolor="#202027"
    )

    st.plotly_chart(fig, use_container_width=True)


with col2:
    top_rated = movies[movies["rating_count"] >= 20].sort_values(
        "average_rating", ascending=False
    ).head(10)

    fig2 = px.bar(
        top_rated.sort_values("average_rating"),
        x="average_rating",
        y="title",
        orientation="h",
        title="Top 10 Highest Rated Movies",
        labels={
            "average_rating": "Average Rating",
            "title": "Movie"
        }
    )

    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101014",
        plot_bgcolor="#202027",
        xaxis_range=[0, 5]
    )

    st.plotly_chart(fig2, use_container_width=True)


# ---------------- GENRE ANALYSIS ----------------
st.markdown(
    '<div class="section-title">🎭 Genre Analysis</div>',
    unsafe_allow_html=True
)

genre_list = []

for genres in movies["genres"].dropna():
    for genre in str(genres).split("|"):
        if genre != "(no genres listed)":
            genre_list.append(genre)

genre_counts = pd.Series(genre_list).value_counts().reset_index()
genre_counts.columns = ["Genre", "Movie Count"]

col3, col4 = st.columns(2)

with col3:
    fig3 = px.pie(
        genre_counts,
        names="Genre",
        values="Movie Count",
        title="Movies by Genre",
        hole=0.4
    )

    fig3.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101014"
    )

    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.bar(
        genre_counts.head(10),
        x="Genre",
        y="Movie Count",
        title="Top 10 Genres"
    )

    fig4.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101014",
        plot_bgcolor="#202027"
    )

    st.plotly_chart(fig4, use_container_width=True)


# ---------------- RATING DISTRIBUTION ----------------
st.markdown(
    '<div class="section-title">⭐ Rating Distribution</div>',
    unsafe_allow_html=True
)

rating_distribution = ratings["rating"].value_counts().sort_index()

fig5 = px.bar(
    x=rating_distribution.index.astype(str),
    y=rating_distribution.values,
    labels={
        "x": "Rating",
        "y": "Number of Ratings"
    },
    title="Distribution of User Ratings"
)

fig5.update_layout(
    template="plotly_dark",
    paper_bgcolor="#101014",
    plot_bgcolor="#202027"
)

st.plotly_chart(fig5, use_container_width=True)


# ---------------- RECENT RATINGS ----------------
st.markdown(
    '<div class="section-title">📝 Recent Rating Records</div>',
    unsafe_allow_html=True
)

recent_ratings = ratings.tail(15).sort_index(ascending=False)

recent_data = recent_ratings.merge(
    movies[["movieId", "title"]],
    on="movieId",
    how="left"
)

st.dataframe(
    recent_data[["userId", "title", "rating"]],
    use_container_width=True,
    hide_index=True
)


# ---------------- MOVIE DATA TABLE ----------------
st.markdown(
    '<div class="section-title">🗂️ Movie Database Overview</div>',
    unsafe_allow_html=True
)

st.dataframe(
    movies[
        ["movieId", "title", "genres", "average_rating", "rating_count"]
    ].sort_values("rating_count", ascending=False),
    use_container_width=True,
    hide_index=True
)


# ---------------- EXPORT ----------------
st.divider()

st.download_button(
    "⬇️ Download Movie Report",
    data=movies.to_csv(index=False).encode("utf-8"),
    file_name="moviemind_admin_report.csv",
    mime="text/csv"
)


# ---------------- FOOTER ----------------
st.divider()

st.caption("MovieMind 2.0 | Administrator Dashboard")
