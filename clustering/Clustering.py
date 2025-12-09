import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import os

# =========================
# 1. Caricamento dati
# =========================
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")

train = pd.read_csv(os.path.join(DATA, "drugLibTrain_final_v4.tsv"), sep="\t")
test  = pd.read_csv(os.path.join(DATA, "drugLibTest_final_v4.tsv"), sep="\t")
df = pd.concat([train, test], ignore_index=True)

# =========================
# 2. Filtra per condizione
# =========================
condition = "depression"
df_cond = df[df["condition_standardized"] == condition]
print(f"Recensioni trovate per '{condition}': {len(df_cond)}")

# =========================
# 3. Mappe numeriche per effectiveness e side_effects
# =========================
effectiveness_map = {
    "Ineffective": 1,
    "Moderately Effective": 2,
    "Considerably Effective": 3,
    "Highly Effective": 4
}

side_effects_map = {
    "None": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3
}

df_cond = df[df["condition_standardized"] == condition].copy()
df_cond.loc[:, "effectiveness_num"] = df_cond["effectiveness"].map(effectiveness_map)
df_cond.loc[:, "side_effects_num"] = df_cond["sideEffects"].map(side_effects_map)

# =========================
# 4. Calcolo Bayesian rating
# =========================
C = df_cond["rating"].mean()
m = 10
review_counts = df_cond.groupby("urlDrugName")["rating"].count()
rating_means = df_cond.groupby("urlDrugName")["rating"].mean()

bayesian_rating = (review_counts / (review_counts + m)) * rating_means + \
                  (m / (review_counts + m)) * C

df_bayes = pd.DataFrame({
    "rating_mean": rating_means,
    "n_reviews": review_counts,
    "bayes_mean": bayesian_rating
})

# =========================
# 5. Calcolo effectiveness_mean e side_effects_mean
# =========================
cluster_df = df_cond.groupby("urlDrugName").agg({
    "rating": ["std", "count"],
    "effectiveness_num": "mean",
    "side_effects_num": "mean"
})

cluster_df.columns = ["rating_std", "n_reviews", "effectiveness_mean", "side_effects_mean"]

# Aggiungo bayes_mean e rating_mean
cluster_df["bayes_mean"] = df_bayes["bayes_mean"]
cluster_df["rating_mean"] = df_bayes["rating_mean"]

cluster_df = cluster_df.fillna(0)

# =========================
# 6. Scaling
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df[["rating_mean", "rating_std", "n_reviews", "bayes_mean",
                                             "effectiveness_mean", "side_effects_mean"]])

# =========================
# 7. K-Prototypes Clustering
# =========================
k = 3
cluster_df['tipo_farmaco'] = df_cond.groupby('urlDrugName')['condition'].first()

X_features = cluster_df[["rating_mean", "rating_std", "n_reviews", "bayes_mean",
                         "effectiveness_mean", "side_effects_mean", "tipo_farmaco"]]
categorical_indices = [6]

kproto = KPrototypes(n_clusters=k, init='Cao', verbose=2, random_state=42)
cluster_df["KPrototypes"] = kproto.fit_predict(X_features.to_numpy(), categorical=categorical_indices)

# =========================
# 8. PCA 2D per visualizzazione
# =========================
pca = PCA(n_components=2)
cluster_df[["PC1", "PC2"]] = pca.fit_transform(X_features.iloc[:, :-1])

plt.figure(figsize=(10, 6))
palette = sns.color_palette("Set1", n_colors=k)
for label in range(k):
    subset = cluster_df[cluster_df["KPrototypes"] == label]
    plt.scatter(subset["PC1"], subset["PC2"], s=subset["n_reviews"]*5, label=f'Cluster {label}')
plt.title(f"K-Prototypes Clustering – {condition}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()

# =========================
# 9. Agglomerative Clustering
# =========================
agglo = AgglomerativeClustering(n_clusters=k)
cluster_df["Agglo"] = agglo.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
for label in range(k):
    subset = cluster_df[cluster_df["Agglo"] == label]
    plt.scatter(subset["PC1"], subset["PC2"], s=subset["n_reviews"]*5, label=f'Cluster {label}')
plt.title(f"Agglomerative Clustering – {condition}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()

# Dendrogramma
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linked, labels=cluster_df.index, leaf_rotation=90)
plt.title(f"Dendrogramma – Hierarchical Clustering ({condition})")
plt.xlabel("Farmaci")
plt.ylabel("Distanza")
plt.tight_layout()
plt.show()

# =========================
# 10. Stampa informazioni sui cluster
# =========================
cluster_order = cluster_df.groupby("KPrototypes")["bayes_mean"].mean().sort_values(ascending=False)
cluster_mapping = {num: name for num, name in zip(cluster_order.index, ["Top", "Medio", "Basso"])}
cluster_df["cluster_name"] = cluster_df["KPrototypes"].map(cluster_mapping)

print("\n=== CLUSTERING FARMACI – Condizione:", condition, "===\n")
print("Feature usate per il clustering:")
print(cluster_df[["rating_mean", "rating_std", "n_reviews", "bayes_mean",
                  "effectiveness_mean", "side_effects_mean"]].head(), "\n")

print("Cluster assegnati con nomi significativi:")
print(cluster_df[["rating_mean", "bayes_mean", "effectiveness_mean", "side_effects_mean", "cluster_name"]].sort_values("bayes_mean", ascending=False))
