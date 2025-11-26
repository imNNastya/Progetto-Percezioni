import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import umap

# =========================
# 1. Caricamento dati
# =========================
train = pd.read_csv("drugLibTrain_raw.tsv", sep="\t")
test = pd.read_csv("drugLibTest_raw.tsv", sep="\t")
df = pd.concat([train, test], ignore_index=True)

# =========================
# 2. Filtra per condizione
# =========================
condition = "anxiety"
df_cond = df[df["condition"] == condition]
print(f"Recensioni trovate per '{condition}': {len(df_cond)}")

# ==========================================================
# 3B. BAYESIAN RATING (INTEGRATO)
# ==========================================================
print("\n=== Calcolo Bayesian Rating ===")

# Media globale della condizione
C = df_cond["rating"].mean()

# Smoothing (puoi aumentarlo se hai tanti farmaci con 1 review)
m = 10

# Conta le recensioni per farmaco
review_counts = df_cond.groupby("urlDrugName")["rating"].count()
rating_means = df_cond.groupby("urlDrugName")["rating"].mean()

bayesian_rating = (review_counts / (review_counts + m)) * rating_means + \
                  (m / (review_counts + m)) * C

df_bayes = pd.DataFrame({
    "rating_mean": rating_means,
    "n_reviews": review_counts,
    "bayes_mean": bayesian_rating
}).sort_values("bayes_mean", ascending=False)

print("\nTop 10 farmaci (Bayesian Ranking):")
print(df_bayes.head(10))

top_drugs = df_bayes.head(10).index.tolist()

print("\nFarmaci penalizzati (rating alto ma poche recensioni):")
df_bayes["delta"] = df_bayes["bayes_mean"] - df_bayes["rating_mean"]
print(df_bayes.sort_values("delta").head(5))

# =========================
# 4. Visualizzazioni preliminari
# =========================
# Lollipop Plot
plt.figure(figsize=(10, 6))
top = df_bayes["bayes_mean"].head(10)
winner = top.index[0]
for i, drug in enumerate(top.index):
    value = top[drug]
    color = "#1f77b4" if drug == winner else "#aec7e8"
    lw = 4 if drug == winner else 2
    size = 600 if drug == winner else 400
    plt.hlines(y=drug, xmin=0, xmax=value, color=color, linewidth=lw, alpha=0.8)
    plt.scatter(value, drug, color=color, s=size, edgecolor='white', linewidth=1.5, zorder=5)
    plt.text(value, i, f"{value:.2f}", va='center', ha='center', fontsize=7, color='white', fontweight='bold', zorder=6)

plt.title(f"Lollipop Plot – {condition} (Bayesian Rating)", fontsize=16)
plt.xlabel("Bayesian Mean")
plt.ylabel("Farmaco")
plt.xlim(0, top.max() + 1)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# 5. Prepara dati per clustering
# =========================
cluster_df = df_cond.groupby("urlDrugName").agg({
    "rating": ["mean", "std", "count"],
    "effectiveness": "count"
})

cluster_df.columns = ["rating_mean", "rating_std", "n_reviews", "eff_count"]
cluster_df["bayes_mean"] = df_bayes["bayes_mean"]  # <--- aggiunto
cluster_df = cluster_df.fillna(0)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df)

# =========================
# 6. K-Means Clustering
# =========================
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_df["KMeans"] = kmeans.fit_predict(X_scaled)

# PCA 2D
pca = PCA(n_components=2)
cluster_df[["PC1", "PC2"]] = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
palette = sns.color_palette("Set1", n_colors=k)
for label in range(k):
    subset = cluster_df[cluster_df["KMeans"] == label]
    plt.scatter(subset["PC1"], subset["PC2"], s=subset["n_reviews"]*5, label=f'Cluster {label}')
plt.title(f"K-Means Clustering – {condition}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()

# =========================
# 7. Agglomerative Clustering
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
# 8. t-SNE & UMAP
# =========================
X_features = cluster_df[['rating_mean', 'rating_std', 'eff_count', 'n_reviews', 'bayes_mean']]

# t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
cluster_df[['TSNE1', 'TSNE2']] = tsne.fit_transform(X_features)

plt.figure(figsize=(12, 8))
sns.scatterplot(data=cluster_df, x='TSNE1', y='TSNE2', hue='KMeans', palette='Set2', s=120)
for idx, row in cluster_df.iterrows():
    plt.text(row.TSNE1 + 0.2, row.TSNE2 + 0.2, idx, fontsize=8)
plt.title("t-SNE Clustering Visualization")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# UMAP
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='euclidean', random_state=42)
cluster_df[['UMAP1', 'UMAP2']] = reducer.fit_transform(X_features)

plt.figure(figsize=(12, 8))
sns.scatterplot(data=cluster_df, x='UMAP1', y='UMAP2', hue='KMeans', palette='Set2', s=120)
for idx, row in cluster_df.iterrows():
    plt.text(row.UMAP1 + 0.2, row.UMAP2 + 0.2, idx, fontsize=8)
plt.title("UMAP Clustering Visualization")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# =========================
# 9. Stampa informazioni utili clustering
# =========================
cluster_order = cluster_df.groupby("KMeans")["bayes_mean"].mean().sort_values(ascending=False)
cluster_mapping = {num: name for num, name in zip(cluster_order.index, ["Top", "Medio", "Basso"])}
cluster_df["cluster_name"] = cluster_df["KMeans"].map(cluster_mapping)

print("\n=== CLUSTERING FARMACI – Condizione:", condition, "===\n")
print("Feature usate per il clustering:")
print(cluster_df[["rating_mean", "rating_std", "n_reviews", "eff_count", "bayes_mean"]].head(), "\n")

print("Cluster assegnati con nomi significativi:")
print(cluster_df[["rating_mean", "bayes_mean", "eff_count", "n_reviews", "cluster_name"]].sort_values("bayes_mean", ascending=False))

