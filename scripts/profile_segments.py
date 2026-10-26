
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

rfm = pd.read_csv(OUTPUT_DIR / "rfm.csv")
features = ["recency", "frequency", "monetary"]
X = rfm[features]

# Use the same scaling and K-means settings as Step 2
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)
rfm["segment_id"] = kmeans.fit_predict(X_scaled)

# Summarise each segment
profiles = (
    rfm.groupby("segment_id")
    .agg(
        customer_count=("customer_id", "size"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    .reset_index()
)

# Give segments interpretable names based on their profiles.
# The highest-mean-monetary segment is always Premium.
premium_id = profiles.loc[
    profiles["mean_monetary"].idxmax(), "segment_id"
]

remaining = profiles.loc[
    profiles["segment_id"] != premium_id
].copy()

frequent_id = remaining.loc[
    remaining["mean_frequency"].idxmax(), "segment_id"
]

remaining = remaining.loc[
    remaining["segment_id"] != frequent_id
].copy()

at_risk_id = remaining.loc[
    remaining["mean_recency"].idxmax(), "segment_id"
]

remaining_id = remaining.loc[
    remaining["segment_id"] != at_risk_id, "segment_id"
].iloc[0]

segment_names = {
    premium_id: "Premium",
    frequent_id: "Frequent Loyal",
    at_risk_id: "At-Risk",
    remaining_id: "Growth Opportunity",
}

recommended_actions = {
    "Premium": "Offer personalised rewards and exclusive benefits.",
    "Frequent Loyal": "Encourage repeat orders with loyalty rewards.",
    "At-Risk": "Send a targeted win-back offer or reminder.",
    "Growth Opportunity": "Use relevant recommendations to encourage repeat purchases.",
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

profiles["segment_name"] = profiles["segment_id"].map(segment_names)
profiles["recommended_action"] = profiles["segment_name"].map(
    recommended_actions
)
profiles["is_premium"] = profiles["segment_id"].eq(premium_id)

# PCA on the scaled RFM matrix
pca = PCA(n_components=2)
pca.fit(X_scaled)
variance = pca.explained_variance_ratio_

# Save segment assignments
segment_output = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]
].copy()

segment_output.to_csv(
    OUTPUT_DIR / "customer_segments.csv",
    index=False
)

profiles.to_csv(
    OUTPUT_DIR / "segment_profiles.csv",
    index=False
)

print("=" * 70)
print("CUSTOMER SEGMENT PROFILES — K=4")
print("=" * 70)
print(
    profiles[
        [
            "segment_id",
            "segment_name",
            "customer_count",
            "mean_recency",
            "mean_frequency",
            "mean_monetary",
            "is_premium",
        ]
    ].to_string(
        index=False,
        formatters={
            "mean_recency": "{:.2f}".format,
            "mean_frequency": "{:.2f}".format,
            "mean_monetary": "{:.2f}".format,
        }
    )
)

print("\nRECOMMENDED BUSINESS ACTIONS")
for _, row in profiles.iterrows():
    print(f"{row['segment_name']}: {row['recommended_action']}")

print("\nPCA EXPLAINED VARIANCE")
print(f"PC1: {variance[0]:.3f}")
print(f"PC2: {variance[1]:.3f}")
print(f"Total: {variance.sum():.3f}")

print("\nPremium segment ID:", premium_id)
print("Premium customer count:", int(rfm["is_premium"].sum()))

# Required output checks
assert len(rfm) == 1175
assert rfm["customer_id"].is_unique
assert rfm["segment_name"].notna().all()
assert rfm["is_premium"].sum() == profiles.loc[
    profiles["is_premium"], "customer_count"
].sum()
assert profiles["is_premium"].sum() == 1
assert segment_output.columns.tolist() == [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium",
]

print("\nSaved outputs:")
print(OUTPUT_DIR / "customer_segments.csv")
print(OUTPUT_DIR / "segment_profiles.csv")
print("SEGMENT PROFILING COMPLETED.")

