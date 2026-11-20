
import streamlit as st
import pandas as pd
import os


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Advanced Recommendations",
    page_icon="⭐",
    layout="wide"
)

st.title("⭐ Advanced Recommendations")

st.write(
    "Explore popular movies and hybrid recommendations."
)

st.markdown("---")


# ==========================================
# LOAD GENERATED RESULTS
# ==========================================

@st.cache_data
def load_csv(path):

    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame()


popular_movies = load_csv(
    "data/processed/popular_movies.csv"
)

hybrid_movies = load_csv(
    "data/processed/improved_hybrid_recommendations.csv"
)


# ==========================================
# SELECT RECOMMENDATION METHOD
# ==========================================

method = st.selectbox(
    "Choose Recommendation Method",
    [
        "Popularity-Based Recommendation",
        "Hybrid Recommendation"
    ]
)


# ==========================================
# POPULARITY RECOMMENDATION
# ==========================================

if method == "Popularity-Based Recommendation":

    st.subheader("🔥 Most Popular Movies")

    if popular_movies.empty:

        st.warning(
            "Popularity results not found. "
            "Run src.popularity_recommendation first."
        )

    else:

        st.success(
            f"Loaded {len(popular_movies)} popular movie records."
        )

        st.dataframe(
            popular_movies.head(50),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Movies are ranked using the weighted rating "
            "calculated by the popularity model."
        )


# ==========================================
# HYBRID RECOMMENDATION
# ==========================================

elif method == "Hybrid Recommendation":

    st.subheader("🎯 Hybrid Movie Recommendations")

    if hybrid_movies.empty:

        st.warning(
            "Hybrid results not found. "
            "Run src.improved_hybrid first."
        )

    else:

        st.success(
            f"Loaded {len(hybrid_movies)} hybrid recommendation records."
        )

        st.dataframe(
            hybrid_movies.head(50),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Hybrid recommendations combine content, "
            "collaborative and popularity signals."
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.info(
    "MovieMind | Advanced Recommendation Engine"
)
