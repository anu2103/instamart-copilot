
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RFM_PATH = ROOT / "outputs" / "rfm.csv"
OUTPUT_DIR = ROOT / "outputs"

rfm = pd.read_csv(RFM_PATH)

features = ["recency", "frequency", "monetary"]
X = rfm[features]

# Standardise the three RFM features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = []

# Evaluate K values from 2 through 8
for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)
    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    results.append({
        "k": k,
        "inertia": inertia,
        "silhouette_score": silhouette
    })

results_df = pd.DataFrame(results)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
results_df.to_csv(
    OUTPUT_DIR / "kmeans_metrics.csv",
    index=False
)

print("=" * 65)
print("K-MEANS CLUSTER EVALUATION")
print("=" * 65)
print(results_df.to_string(
    index=False,
    formatters={
        "inertia": "{:.1f}".format,
        "silhouette_score": "{:.3f}".format
    }
))

print("\nInterpretation guide:")
print("- Inertia measures how tightly customers fit within clusters.")
print("- Lower inertia is expected as K increases.")
print("- Silhouette measures how well-separated the clusters are.")
print("- Consider the elbow, silhouette scores and business usefulness.")

print("\nSaved to:", OUTPUT_DIR / "kmeans_metrics.csv")
print("CLUSTER EVALUATION COMPLETED.")
