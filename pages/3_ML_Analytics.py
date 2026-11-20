
import streamlit as st
import pandas as pd
import os


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ML Analytics | MovieMind",
    page_icon="📊",
    layout="wide"
)

st.title("📊 MovieMind - ML Analytics Dashboard")

st.write(
    "Explore model performance, recommendation metrics "
    "and unsupervised learning results."
)

st.markdown("---")


# ==========================================
# HELPER FUNCTIONS
# ==========================================

DATA_DIR = "data/processed"


def load_csv(filename):

    path = os.path.join(DATA_DIR, filename)

    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as error:
            st.warning(f"Could not read {filename}: {error}")

    return pd.DataFrame()


def show_csv_section(title, filename):

    st.subheader(title)

    data = load_csv(filename)

    if data.empty:
        st.info(f"{filename} not found or contains no records.")
        return

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )

    numeric_columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:
        st.caption("Numeric data preview")
        st.bar_chart(data[numeric_columns].head(20))


def show_image(title, filename):

    st.subheader(title)

    path = os.path.join(DATA_DIR, filename)

    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.info(f"{filename} not generated yet.")


# ==========================================
# DASHBOARD NAVIGATION
# ==========================================

section = st.selectbox(
    "Select Analytics Section",
    [
        "Overview",
        "Regression",
        "Classification",
        "Clustering",
        "PCA Visualization",
        "Recommendation Evaluation"
    ]
)


# ==========================================
# OVERVIEW
# ==========================================

if section == "Overview":

    st.header("Project Overview")

    movies = load_csv("movies_clean.csv")
    ratings = load_csv("ratings_clean.csv")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Movies",
            f"{len(movies):,}" if not movies.empty else "N/A"
        )

    with col2:
        st.metric(
            "Total Ratings",
            f"{len(ratings):,}" if not ratings.empty else "N/A"
        )

    with col3:
        st.metric(
            "ML Algorithms",
            "8+"
        )

    st.markdown("---")

    st.subheader("Model Comparison")

    comparison = load_csv("model_comparison.csv")

    if comparison.empty:

        st.info("Model comparison file is not available.")

    else:

        if "Score" in comparison.columns:

            comparison["Score"] = pd.to_numeric(
                comparison["Score"],
                errors="coerce"
            )

            available = comparison.dropna(
                subset=["Score"]
            )

            if available.empty:
                st.info(
                    "No actual model scores entered yet. "
                    "Run the model scripts and add their real metrics."
                )
            else:
                st.dataframe(
                    available,
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )


# ==========================================
# REGRESSION
# ==========================================

elif section == "Regression":

    st.header("📈 Regression Model Results")

    st.write(
        "Linear Regression is used to predict movie ratings."
    )

    show_image(
        "Linear Regression Evaluation",
        "linear_regression_results.png"
    )

    st.info(
        "Regression metrics can also be added to the model "
        "comparison CSV after collecting actual script outputs."
    )


# ==========================================
# CLASSIFICATION
# ==========================================

elif section == "Classification":

    st.header("🤖 Classification Model Results")

    st.write(
        "Logistic Regression, Naive Bayes and SVM classification."
    )

    tab1, tab2, tab3 = st.tabs([
        "Logistic Regression",
        "Naive Bayes",
        "SVM"
    ])

    with tab1:
        show_image(
            "Logistic Regression Confusion Matrix",
            "logistic_confusion_matrix.png"
        )

    with tab2:
        show_image(
            "Naive Bayes Confusion Matrix",
            "naive_bayes_confusion_matrix.png"
        )

    with tab3:
        show_image(
            "SVM Confusion Matrix",
            "svm_confusion_matrix.png"
        )

    st.caption(
        "Classification task: movie like/dislike or genre "
        "classification, depending on the model."
    )


# ==========================================
# CLUSTERING
# ==========================================

elif section == "Clustering":

    st.header("🔍 Unsupervised Learning Results")

    show_csv_section(
        "K-Means Movie Clusters",
        "kmeans_movie_clusters.csv"
    )

    show_image(
        "K-Means Cluster Visualization",
        "kmeans_clusters.png"
    )

    st.markdown("---")

    show_csv_section(
        "EM / Gaussian Mixture Model Clusters",
        "em_movie_clusters.csv"
    )

    show_image(
        "EM Cluster Visualization",
        "em_clusters.png"
    )

    st.markdown("---")

    show_csv_section(
        "Clustering Model Comparison",
        "clustering_comparison.csv"
    )


# ==========================================
# PCA
# ==========================================

elif section == "PCA Visualization":

    st.header("📉 Principal Component Analysis")

    show_image(
        "PCA Movie Visualization",
        "pca_visualization.png"
    )

    show_csv_section(
        "PCA Movie Data",
        "pca_movie_data.csv"
    )

    st.write(
        "PCA reduces multiple movie features into two "
        "principal components for visualization."
    )


# ==========================================
# RECOMMENDATION EVALUATION
# ==========================================

elif section == "Recommendation Evaluation":

    st.header("🎯 Recommendation Model Evaluation")

    show_csv_section(
        "Final Recommendation Comparison",
        "final_recommendation_results.csv"
    )

    show_csv_section(
        "Earlier Evaluation Results",
        "evaluation_results.csv"
    )

    st.write(
        "Metrics include Precision@10, Recall@10 and F1@10 "
        "where available."
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "MovieMind | Machine Learning Analytics Dashboard"
)
