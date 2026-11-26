
import streamlit as st
import pandas as pd
import os
from sqlalchemy import create_engine, text

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SQL Analytics | MovieMind",
    page_icon="🗄️",
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
.sql-card {
    background: #202027;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #33333D;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- DATABASE CONNECTION ----------------
DATABASE_URL = st.secrets.get("DATABASE_URL", "")

engine = None

if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        db_connected = True
    except Exception as e:
        db_connected = False
        st.warning(f"Database connection failed: {e}")
else:
    db_connected = False


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎬 MovieMind")
    st.caption("SQL Analytics")

    st.divider()

    st.page_link("app.py", label="Main Dashboard", icon="🏠")
    st.page_link("pages/1_Home.py", label="Home", icon="🎬")
    st.page_link("pages/2_Movie_Search.py", label="Movie Search", icon="🔍")
    st.page_link("pages/3_Admin_Dashboard.py", label="Admin Dashboard", icon="📊")
    st.page_link("pages/4_SQL_Analytics.py", label="SQL Analytics", icon="🗄️")


# ---------------- HEADER ----------------
st.markdown(
    '<div class="page-title">🗄️ SQL Analytics</div>',
    unsafe_allow_html=True
)

st.write("Explore movie data using database queries and SQL analytics.")

if db_connected:
    st.success("PostgreSQL database connected")
else:
    st.info("PostgreSQL is not connected yet. CSV preview mode is active.")


# ---------------- DATA LOADING ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MOVIES_PATH = os.path.join(
    BASE_DIR, "data", "processed", "movies_clean.csv"
)

RATINGS_PATH = os.path.join(
    BASE_DIR, "data", "processed", "ratings_clean.csv"
)

@st.cache_data
def load_csv_data():
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    return movies, ratings


try:
    csv_movies, csv_ratings = load_csv_data()
except FileNotFoundError:
    csv_movies, csv_ratings = pd.DataFrame(), pd.DataFrame()


def run_query(query, fallback):
    if db_connected:
        try:
            with engine.connect() as connection:
                return pd.read_sql(text(query), connection)
        except Exception as e:
            st.error(f"SQL query error: {e}")

    return fallback


# ---------------- SQL QUERY EXAMPLES ----------------
st.divider()
st.subheader("📌 SQL Query Demonstrations")

query_options = {
    "Total Movies": """
        SELECT COUNT(*) AS total_movies
        FROM movies
    """,

    "Average Movie Rating": """
        SELECT ROUND(AVG(rating)::numeric, 2) AS average_rating
        FROM ratings
    """,

    "Top 10 Popular Movies": """
        SELECT m.title, COUNT(r.rating_id) AS total_ratings
        FROM movies m
        JOIN ratings r ON m.movie_id = r.movie_id
        GROUP BY m.movie_id, m.title
        ORDER BY total_ratings DESC
        LIMIT 10
    """,

    "Highest Rated Movies": """
        SELECT m.title,
               ROUND(AVG(r.rating)::numeric, 2) AS average_rating,
               COUNT(r.rating_id) AS total_ratings
        FROM movies m
        JOIN ratings r ON m.movie_id = r.movie_id
        GROUP BY m.movie_id, m.title
        HAVING COUNT(r.rating_id) >= 20
        ORDER BY average_rating DESC
        LIMIT 10
    """,

    "Most Active Users": """
        SELECT u.username, COUNT(r.rating_id) AS total_ratings
        FROM users u
        JOIN ratings r ON u.user_id = r.user_id
        GROUP BY u.user_id, u.username
        ORDER BY total_ratings DESC
        LIMIT 10
    """,

    "Genre-wise Movie Count": """
        SELECT g.genre_name, COUNT(mg.movie_id) AS movie_count
        FROM genres g
        JOIN movie_genres mg ON g.genre_id = mg.genre_id
        GROUP BY g.genre_id, g.genre_name
        ORDER BY movie_count DESC
    """
}

selected_query = st.selectbox(
    "Select SQL Analysis",
    list(query_options.keys())
)

st.markdown("#### SQL Query")

st.code(query_options[selected_query], language="sql")


# ---------------- FALLBACK DATA ----------------
def fallback_query(name):
    if csv_movies.empty:
        return pd.DataFrame()

    stats = csv_ratings.groupby("movieId").agg(
        total_ratings=("rating", "count"),
        average_rating=("rating", "mean")
    ).reset_index()

    if name == "Total Movies":
        return pd.DataFrame({"total_movies": [len(csv_movies)]})

    if name == "Average Movie Rating":
        return pd.DataFrame({
            "average_rating": [round(csv_ratings["rating"].mean(), 2)]
        })

    if name == "Top 10 Popular Movies":
        data = csv_movies.merge(stats, on="movieId", how="left")
        data = data.sort_values("total_ratings", ascending=False).head(10)
        return data[["title", "total_ratings"]]

    if name == "Highest Rated Movies":
        data = csv_movies.merge(stats, on="movieId", how="left")
        data = data[data["total_ratings"] >= 20]
        data = data.sort_values("average_rating", ascending=False).head(10)
        return data[["title", "average_rating", "total_ratings"]]

    if name == "Most Active Users":
        users = csv_ratings.groupby("userId").size().reset_index(
            name="total_ratings"
        )
        users = users.sort_values("total_ratings", ascending=False).head(10)
        users["username"] = users["userId"].apply(
            lambda x: f"Demo User {x}"
        )
        return users[["username", "total_ratings"]]

    if name == "Genre-wise Movie Count":
        genre_list = []

        for genres in csv_movies["genres"].dropna():
            for genre in str(genres).split("|"):
                if genre != "(no genres listed)":
                    genre_list.append(genre)

        return pd.Series(genre_list).value_counts().reset_index().rename(
            columns={"index": "genre_name", "count": "movie_count"}
        )

    return pd.DataFrame()


# ---------------- EXECUTE QUERY ----------------
if st.button("▶ Run SQL Query", type="primary"):
    if db_connected:
        result = run_query(
            query_options[selected_query],
            pd.DataFrame()
        )
    else:
        result = fallback_query(selected_query)

    st.session_state["sql_result"] = result

if "sql_result" in st.session_state:
    st.markdown("#### 📊 Query Result")

    result = st.session_state["sql_result"]

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Query Result",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="sql_analytics_result.csv",
        mime="text/csv"
    )


# ---------------- SQL CONCEPTS ----------------
st.divider()
st.subheader("📚 SQL Concepts Demonstrated")

concepts = [
    ("COUNT()", "Counts database records"),
    ("AVG()", "Calculates average rating"),
    ("INNER JOIN", "Combines related tables"),
    ("GROUP BY", "Groups records for analysis"),
    ("HAVING", "Filters grouped results"),
    ("ORDER BY", "Sorts query results"),
    ("LIMIT", "Restricts returned records")
]

for concept, description in concepts:
    st.markdown(
        f"""
        <div class="sql-card">
            <b>{concept}</b><br>
            <span style="color:#B8B8C2">{description}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------- FOOTER ----------------
st.divider()
st.caption("MovieMind 2.0 | SQL Analytics Module")
