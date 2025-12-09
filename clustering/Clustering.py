import pandas as pd #Gestione e manipolazione dei dati in formato tabellare (DataFrame), cioè leggere CSV/TSV (pd.read_csv), raggruppare dati (groupby), aggregazioni (agg), unire DataFrame (concat).
import seaborn as sns #usata per la palette di colori (sns.color_palette).
import matplotlib.pyplot as plt #Libreria base per creare grafici
from sklearn.preprocessing import StandardScaler #Serve a standardizzare le feature numeriche, Essenziale prima di clustering o PCA per evitare che scale diverse dominino i risultati.
from sklearn.decomposition import PCA #Riduzione della dimensionalità dei dati numerici. lo usi per trasformare dati multi-dimensionali in 2 componenti principali (PC1 e PC2), così puoi fare scatter plot.
from kmodes.kprototypes import KPrototypes 
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage #linkage: calcola le distanze tra i cluster (usato per il dendrogramma). dendrogram: visualizza la gerarchia dei cluster sotto forma di albero.
import numpy as np # Importato per le operazioni numeriche

#rating_mean -> media del rating del farmaco
#rating_std -> variabilità del rating (quanto i voti sono diversi tra loro)
#n_reviews -> numero di recensioni del farmaco
#eff_count -> quante recensioni riportano l’attributo “effectiveness”

# =========================
# 1. Caricamento dati. 
# =========================
#Leggi i dati TSV (tab-separated values) di train e test
train = pd.read_csv("drugLibTrain_final_v4.tsv", sep="\t")
test = pd.read_csv("drugLibTest_final_v4.tsv", sep="\t")
df = pd.concat([train, test], ignore_index=True) #pd.concat li unisce in un unico DataFrame (df).

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
cluster_df[["Successo Complessivo del Farmaco", "Accordo/Disaccordo degli Utenti"]] = pca.fit_transform(X_features.iloc[:, :-1])

# Nomi personalizzati dei cluster
cluster_names = {
    0: "Top",
    1: "Medio",
    2: "Basso"
}
plt.figure(figsize=(10, 6))
palette = sns.color_palette("Set1", n_colors=k)
for label in range(k):
    subset = cluster_df[cluster_df["KPrototypes"] == label]
    plt.scatter(subset["Successo Complessivo del Farmaco"], subset["Accordo/Disaccordo degli Utenti"], s=subset["n_reviews"]*5, label=cluster_names[label]) 
plt.title(f"K-Prototypes Clustering – {condition}")
plt.xlabel("Successo Complessivo del Farmaco")
plt.ylabel("Accordo/Disaccordo degli Utenti")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()

# =========================
# 9. Agglomerative Clustering
# =========================
agglo = AgglomerativeClustering(n_clusters=k)
cluster_df["Agglo"] = agglo.fit_predict(X_scaled)

cluster_names = {
    0: "Top",
    1: "Basso",
    2: "Medio"
}

plt.figure(figsize=(10, 6))
for label in range(k):
    subset = cluster_df[cluster_df["Agglo"] == label]
    plt.scatter(subset["Successo Complessivo del Farmaco"], subset["Accordo/Disaccordo degli Utenti"], s=subset["n_reviews"]*5, label=cluster_names[label])
plt.title(f"Agglomerative Clustering – {condition}")
plt.xlabel("Successo Complessivo del Farmaco")
plt.ylabel("Accordo/Disaccordo degli Utenti")
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
