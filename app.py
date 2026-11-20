
import streamlit as st
import pandas as pd
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MovieMind",
    page_icon="🎬",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #10131c;
    color: white;
}

.main-title {
    font-size: 48px;
    font-weight: bold;
    color: #ff4b6e;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #b8bdc9;
}

.movie-card {
    background-color: #202635;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_movies():

    path = "data/processed/movies_clean.csv"

    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame()


movies = load_movies()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🎬 MovieMind")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Movie Explorer",
        "About Project"
    ]
)


# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.markdown(
        '<div class="main-title">🎬 MovieMind</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Your Intelligent Movie Recommendation System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("Welcome to MovieMind!")

    st.write(
        "MovieMind is an intelligent movie recommendation "
        "system that uses Machine Learning algorithms "
        "to discover movies based on user preferences."
    )

    # Dataset statistics

    if not movies.empty:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Movies",
                f"{len(movies):,}"
            )

        with col2:
            st.metric(
                "Genres",
                movies["genres"].nunique()
            )

        with col3:
            st.metric(
                "Dataset",
                "MovieLens"
            )

    st.markdown("---")

    st.subheader("What MovieMind Offers")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        ### 🎯 Personalized Recommendations

        - Content-Based Filtering
        - Collaborative Filtering
        - Hybrid Recommendation
        - Popularity-Based Recommendation
        """)

    with col2:

        st.markdown("""
        ### 🤖 Machine Learning

        - Linear Regression
        - Logistic Regression
        - Naive Bayes
        - Support Vector Machine
        - K-Means and PCA
        - EM and MDP
        """)

    st.markdown("---")

    st.success(
        "Explore movies and discover your next favourite film!"
    )


# ==========================================
# MOVIE EXPLORER
# ==========================================

elif page == "Movie Explorer":

    st.title("🔍 Movie Explorer")

    if movies.empty:

        st.error("Movie dataset not found!")

    else:

        search = st.text_input(
            "Search for a movie",
            placeholder="Enter movie name..."
        )

        if search:

            results = movies[
                movies["title"].str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

            st.write(
                f"Found {len(results)} matching movies."
            )

            for _, movie in results.head(20).iterrows():

                st.markdown(
                    f"""
                    <div class="movie-card">
                        <h4>{movie['title']}</h4>
                        <p>Genres: {movie['genres']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "Enter a movie title above to explore movies."
            )


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.title("📚 About MovieMind")

    st.write("""
    MovieMind is a Machine Learning based
    Intelligent Movie Recommendation System.

    Dataset:
    MovieLens Latest Small

    Technologies Used:
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Streamlit
    - Matplotlib
    - Plotly

    Recommendation Techniques:
    - Content-Based Filtering
    - Collaborative Filtering
    - Hybrid Filtering
    - Popularity-Based Filtering

    Machine Learning Algorithms:
    - Linear Regression
    - Logistic Regression
    - Naive Bayes
    - SVM
    - K-Means
    - PCA
    - Gaussian Mixture Model
    - MDP
    """)

    st.markdown("---")

    st.info(
        "MovieMind - Intelligent Movie Recommendation System"
    )
