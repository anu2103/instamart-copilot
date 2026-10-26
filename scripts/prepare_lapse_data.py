
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "lapse_dataset.csv"

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

# Stratified split preserves the approximate class proportion.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("=" * 65)
print("TRAIN/TEST SPLIT")
print("=" * 65)
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Lapsed in training:", int(y_train.sum()))
print("Not lapsed in training:", int(y_train.eq(0).sum()))
print("Lapsed in test:", int(y_test.sum()))
print("Not lapsed in test:", int(y_test.eq(0).sum()))

# Fit preprocessing on training rows only.
preprocessor = ColumnTransformer(
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

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Verify that preprocessing created no missing values.
train_values = (
    X_train_processed.toarray()
    if hasattr(X_train_processed, "toarray")
    else np.asarray(X_train_processed)
)
test_values = (
    X_test_processed.toarray()
    if hasattr(X_test_processed, "toarray")
    else np.asarray(X_test_processed)
)

assert np.isfinite(train_values).all()
assert np.isfinite(test_values).all()

# Compare the training-fitted frequency mean with the full cohort mean.
numeric_scaler = preprocessor.named_transformers_["numeric"]
frequency_index = numeric_features.index("frequency")

training_scaler_frequency_mean = numeric_scaler.mean_[frequency_index]
full_cohort_frequency_mean = dataset["frequency"].mean()

print("\nFREQUENCY MEAN COMPARISON")
print(
    "Training-only scaler mean:",
    round(training_scaler_frequency_mean, 4),
)
print(
    "Full-cohort frequency mean:",
    round(full_cohort_frequency_mean, 4),
)
print(
    "The means differ because the scaler was fitted only on training "
    "rows, while the full-cohort mean includes the held-out test rows."
)

# Baseline: always predict not lapsed (class 0).
baseline_predictions = np.zeros(len(y_test), dtype=int)

baseline_accuracy = accuracy_score(y_test, baseline_predictions)
baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    pos_label=1,
    zero_division=0,
)
baseline_cm = confusion_matrix(
    y_test,
    baseline_predictions,
    labels=[0, 1],
)

print("\nBASELINE — ALWAYS PREDICT NOT LAPSED")
print("Accuracy:", round(baseline_accuracy, 4))
print("Lapsed-class F1:", round(baseline_f1, 4))
print("Confusion matrix (rows=actual [0,1], columns=predicted [0,1]):")
print(baseline_cm)

print("\nProcessed training matrix shape:", X_train_processed.shape)
print("Processed test matrix shape:", X_test_processed.shape)
print("Preprocessing fitted on training rows only: Yes")
print("Test rows used to fit encoders or scalers: No")
print("Missing or infinite processed values: None")
print("PREPROCESSING CHECKS COMPLETED.")
