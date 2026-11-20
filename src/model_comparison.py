
import pandas as pd
import matplotlib.pyplot as plt
import os


# Enter actual results from your program outputs
results = [
    {
        "Model": "Linear Regression",
        "Task": "Rating Prediction",
        "Metric": "MAE",
        "Score": None
    },
    {
        "Model": "Linear Regression",
        "Task": "Rating Prediction",
        "Metric": "RMSE",
        "Score": None
    },
    {
        "Model": "Logistic Regression",
        "Task": "Like/Dislike Classification",
        "Metric": "Accuracy",
        "Score": None
    },
    {
        "Model": "Naive Bayes",
        "Task": "Genre Classification",
        "Metric": "Accuracy",
        "Score": None
    },
    {
        "Model": "SVM",
        "Task": "Genre Classification",
        "Metric": "Accuracy",
        "Score": None
    }
]


# Convert to DataFrame
df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(df.to_string(index=False))


# Save comparison template
os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)

print("\nComparison CSV saved!")


# Plot only when actual scores are entered
plot_data = df.dropna(subset=["Score"])

if not plot_data.empty:

    plt.figure(figsize=(10, 6))

    plt.bar(
        plot_data["Model"] + " - " + plot_data["Metric"],
        plot_data["Score"]
    )

    plt.title("MovieMind - Model Performance Comparison")
    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    plt.savefig(
        "data/processed/model_comparison.png"
    )

    plt.close()

    print("Comparison graph saved!")

else:
    print(
        "\nNo scores entered yet. "
        "Fill actual metrics before generating the graph."
    )


print("\nMODEL COMPARISON COMPLETED!")
