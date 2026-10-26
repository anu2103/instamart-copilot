
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
DATA_PATH = OUTPUT_DIR / "lapse_dataset.csv"

dataset = pd.read_csv(DATA_PATH)

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

categorical_features = [
    "city",
    "gender",
    "favourite_category",
]

feature_columns = numeric_features + categorical_features

X = dataset[feature_columns]
y = dataset["lapsed"]

# Reproduce the exact split from Step 5.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

# Baseline: predict that every test customer will not lapse.
baseline_predictions = [0] * len(y_test)
print("=" * 70)
print("BASELINE")
print("=" * 70)
print("Accuracy:", round(accuracy_score(y_test, baseline_predictions), 4))
print(
    "Lapsed-class F1:",
    round(f1_score(y_test, baseline_predictions, zero_division=0), 4),
)

# Logistic regression pipeline and grid search.
logistic_pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("model", LogisticRegression(max_iter=1000)),
])

logistic_grid = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid={
        "model__C": [0.01, 0.1, 1, 10, 100],
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
    refit=True,
)

print("\nTuning logistic regression...")
logistic_grid.fit(X_train, y_train)

# Random forest pipeline and grid search.
forest_pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )),
])

forest_grid = GridSearchCV(
    estimator=forest_pipeline,
    param_grid={
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20],
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
    refit=True,
)

print("Tuning random forest (45 fits across 5 folds)...")
forest_grid.fit(X_train, y_train)

def evaluate_model(name, search):
    model = search.best_estimator_
    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    precision = precision_score(
        y_test, predictions, pos_label=1, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label=1, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, pos_label=1, zero_division=0
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print("Best parameters:", search.best_params_)
    print("Best cross-validated F1:", round(search.best_score_, 4))
    print("Test confusion matrix (rows=actual [0,1], columns=predicted [0,1]):")
    print(cm)
    print("Lapsed-class precision:", round(precision, 4))
    print("Lapsed-class recall:   ", round(recall, 4))
    print("Lapsed-class F1:       ", round(f1, 4))

    return {
        "model_name": name,
        "best_params": str(search.best_params_),
        "best_cv_f1": search.best_score_,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "confusion_matrix": str(cm.tolist()),
    }

logistic_result = evaluate_model(
    "LOGISTIC REGRESSION",
    logistic_grid,
)
forest_result = evaluate_model(
    "RANDOM FOREST",
    forest_grid,
)

# Save evaluation results for documentation.
results = pd.DataFrame([logistic_result, forest_result])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_DIR / "model_evaluation.csv", index=False)

# Save all cross-validation results for reproducibility.
pd.DataFrame(logistic_grid.cv_results_).to_csv(
    OUTPUT_DIR / "logistic_cv_results.csv", index=False
)
pd.DataFrame(forest_grid.cv_results_).to_csv(
    OUTPUT_DIR / "forest_cv_results.csv", index=False
)

print("\nSaved evaluation and cross-validation results in outputs/.")
print("MODEL TUNING AND EVALUATION COMPLETED.")
