
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

dataset = pd.read_csv(OUTPUT_DIR / "lapse_dataset.csv")

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]
categorical_features = ["city", "gender", "favourite_category"]
feature_columns = numeric_features + categorical_features

X = dataset[feature_columns]
y = dataset["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("Original training class counts:")
print(y_train.value_counts().sort_index())

# Oversample only the training data.
training_data = X_train.copy()
training_data["lapsed"] = y_train.to_numpy()

majority_count = int(y_train.value_counts().max())
minority_rows = training_data[training_data["lapsed"] == 1]

sampled_lapsed = minority_rows.sample(
    n=majority_count - len(minority_rows),
    replace=True,
    random_state=42,
)

balanced_data = pd.concat(
    [training_data, sampled_lapsed],
    ignore_index=True,
)

X_balanced = balanced_data[feature_columns]
y_balanced = balanced_data["lapsed"]

print("\nBalanced training class counts:")
print(y_balanced.value_counts().sort_index())
print("\nTest rows remain unchanged:", len(X_test))

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

# Reproduce the selected Random Forest model.
# Fit preprocessing on balanced training data only.
balanced_forest = Pipeline(
    steps=[
        ("preprocessor", make_preprocessor()),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=10,
                random_state=42,
            ),
        ),
    ]
)

# Use the same untouched test set for both evaluations.
original_forest = clone(balanced_forest)
original_forest.fit(X_train, y_train)
balanced_forest.fit(X_balanced, y_balanced)

def evaluate(name, model):
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(
        "Confusion matrix [[TN, FP], [FN, TP]]:"
    )
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(
        "Lapsed precision:",
        round(precision_score(y_test, predictions, zero_division=0), 4),
    )
    print(
        "Lapsed recall:",
        round(recall_score(y_test, predictions, zero_division=0), 4),
    )
    print(
        "Lapsed F1:",
        round(f1_score(y_test, predictions, zero_division=0), 4),
    )

evaluate("ORIGINAL TRAINING DATA", original_forest)
evaluate("OVERSAMPLED TRAINING DATA", balanced_forest)

print("\nOVERSAMPLING EXPERIMENT COMPLETED.")
