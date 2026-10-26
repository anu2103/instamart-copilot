
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset = pd.read_csv(OUTPUT_DIR / "lapse_dataset.csv")
segments = pd.read_csv(OUTPUT_DIR / "customer_segments.csv")

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

# Refit the selected Random Forest configuration on original training data.
forest = Pipeline(
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
forest.fit(X_train, y_train)

# 1. Random Forest feature importance
preprocessor = forest.named_steps["preprocessor"]
forest_model = forest.named_steps["model"]
feature_names = preprocessor.get_feature_names_out()
importance = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": forest_model.feature_importances_,
    }
).sort_values("importance", ascending=False)

print("=" * 65)
print("TOP FIVE RANDOM FOREST FEATURE IMPORTANCES")
print("=" * 65)
print(importance.head(5).to_string(index=False))

importance.to_csv(
    OUTPUT_DIR / "random_forest_feature_importance.csv",
    index=False,
)

# 2. Fit Logistic Regression with the best parameters from Task 6.
logistic = Pipeline(
    steps=[
        ("preprocessor", make_preprocessor()),
        ("model", LogisticRegression(C=100, max_iter=1000)),
    ]
)
logistic.fit(X_train, y_train)

logistic_model = logistic.named_steps["model"]
logistic_names = logistic.named_steps[
    "preprocessor"
].get_feature_names_out()

coefficients = pd.DataFrame(
    {
        "feature": logistic_names,
        "coefficient": logistic_model.coef_[0],
    }
).sort_values("coefficient")

print("\n" + "=" * 65)
print("THREE MOST NEGATIVE LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 65)
print(coefficients.head(3).to_string(index=False))

print("\n" + "=" * 65)
print("THREE MOST POSITIVE LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 65)
print(coefficients.tail(3).sort_values(
    "coefficient", ascending=False
).to_string(index=False))

coefficients.to_csv(
    OUTPUT_DIR / "logistic_regression_coefficients.csv",
    index=False,
)

# 3. Save, reload, and verify exact predictions.
model_path = OUTPUT_DIR / "lapse_model.joblib"
joblib.dump(forest, model_path)

reloaded_model = joblib.load(model_path)
predictions_before = forest.predict(X_test)
predictions_after = reloaded_model.predict(X_test)

assert np.array_equal(predictions_before, predictions_after)
print("\nModel saved to:", model_path)
print("Reloaded model predictions match exactly.")

# 4. Build a post-hoc Premium-segment retention shortlist.
# Segment membership is used only after the predictive test split.
premium_ids = set(
    segments.loc[segments["is_premium"].astype(bool), "customer_id"]
)

test_customers = X_test[["recency"]].copy()
test_customers["customer_id"] = X_test.index.map(
    dataset.loc[X_test.index, "customer_id"]
)
test_customers["lapse_probability"] = reloaded_model.predict_proba(
    X_test
)[:, 1]
test_customers["predicted_lapsed"] = predictions_after
test_customers["actually_lapsed"] = y_test.to_numpy()

# Use the original split rows' customer IDs to select Premium customers.
test_customers = test_customers[
    test_customers["customer_id"].isin(premium_ids)
].copy()

retention_list = (
    test_customers.sort_values(
        "lapse_probability", ascending=False
    )
    .head(10)
    [["customer_id", "lapse_probability", "actually_lapsed"]]
)

retention_path = OUTPUT_DIR / "retention_list.csv"
retention_list.to_csv(retention_path, index=False)

print("\n" + "=" * 65)
print("TOP PREMIUM CUSTOMERS BY PREDICTED LAPSE PROBABILITY")
print("=" * 65)
print(retention_list.to_string(index=False))
print("\nRetention list saved to:", retention_path)
print("\nTASK 8 MODEL FINALIZATION COMPLETED.")
