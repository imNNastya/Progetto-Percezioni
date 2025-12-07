import pandas as pd #Gestione e manipolazione dei dati in formato tabellare (DataFrame), cioè leggere CSV/TSV (pd.read_csv), raggruppare dati (groupby), aggregazioni (agg), unire DataFrame (concat).
import seaborn as sns #usata per la palette di colori (sns.color_palette).
import matplotlib.pyplot as plt #Libreria base per creare grafici
from sklearn.preprocessing import StandardScaler #Serve a standardizzare le feature numeriche, Essenziale prima di clustering o PCA per evitare che scale diverse dominino i risultati.
from sklearn.decomposition import PCA #Riduzione della dimensionalità dei dati numerici. lo usi per trasformare dati multi-dimensionali in 2 componenti principali (PC1 e PC2), così puoi fare scatter plot.
from kmodes.kprototypes import KPrototypes 
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage #linkage: calcola le distanze tra i cluster (usato per il dendrogramma). dendrogram: visualizza la gerarchia dei cluster sotto forma di albero.

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
#Selezioni solo le recensioni relative a una condizione specifica (es. anxiety).
condition = "anxiety"
df_cond = df[df["condition"] == condition]
print(f"Recensioni trovate per '{condition}': {len(df_cond)}") #len(df_cond) stampa il numero di recensioni trovate.

# ==========================================================
# 3. BAYESIAN RATING. Aiuta ad ottenere cluster più robusti, realistici e coerenti, perchè si ha la media pesata che tiene conto del numero di reviews (quindi un farmaco con un solo 10/10 non è meglio di un farmaco con 8/10 ma con più recensioni)
# ==========================================================
print("\n=== Calcolo Bayesian Rating ===")

# Media globale della condizione
C = df_cond["rating"].mean()

# Smoothing (si può aumentare se si hanno tanti farmaci con 1 review)
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
# 4. Prepara dati per clustering
# =========================
cluster_df = df_cond.groupby("urlDrugName").agg({ #Costruisce il DataFrame delle feature di ogni farmaco
    "rating": ["mean", "std", "count"],
    "effectiveness": "count"
})

cluster_df.columns = ["rating_mean", "rating_std", "n_reviews", "eff_count"]
cluster_df["bayes_mean"] = df_bayes["bayes_mean"]  # <--- aggiunto
cluster_df = cluster_df.fillna(0) #Riempie eventuali valori mancanti

# Scaling : Standardizza tutte le feature numeriche
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df)

# =========================
# 5. K-Prototypes Clustering. Algoritmo di clustering per dati misti, cioè contenenti variabili numeriche (rating_mean) e categoriche (tipo_farmaco).
# =========================
k = 3

# Aggiungo una colonna categoriale fittizia (tipo farmaco)
cluster_df['tipo_farmaco'] = df_cond.groupby('urlDrugName')['condition'].first()

# Preparo i dati per K-Prototypes
X_features = cluster_df[['rating_mean', 'rating_std', 'eff_count', 'n_reviews', 'bayes_mean', 'tipo_farmaco']]
categorical_indices = [5]  # indice della colonna 'tipo_farmaco'

# Fit K-Prototypes
kproto = KPrototypes(n_clusters=k, init='Cao', verbose=2, random_state=42)
cluster_df["KPrototypes"] = kproto.fit_predict(X_features.to_numpy(), categorical=categorical_indices)

# PCA 2D per visualizzazione, PC1 e PC2 servono solo per ridurre la dimensionalità e visualizzare i cluster in 2D.
pca = PCA(n_components=2)
cluster_df[["Successo Complessivo del Farmaco", "Accordo/Disaccordo degli Utenti"]] = pca.fit_transform(X_features.iloc[:, :-1])  # solo colonne numeriche

# Plot PCA -> viene fatta una sola volta perchè vengono definite le dimensione indipendentemente dagli algoritmi usati (nel senso i dati entrano in input e si fissano le dimensioni, poi lo si fa per ogni algoritmo usato)
plt.figure(figsize=(10, 6))
palette = sns.color_palette("Set1", n_colors=k)
for label in range(k):
    subset = cluster_df[cluster_df["KPrototypes"] == label]
    plt.scatter(subset["Successo Complessivo del Farmaco"], subset["Accordo/Disaccordo degli Utenti"], s=subset["n_reviews"]*5, label=f'Cluster {label}')
plt.title(f"K-Prototypes Clustering – {condition}")
plt.xlabel("Successo Complessivo del Farmaco")
plt.ylabel("Accordo/Disaccordo degli Utenti")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()

# =========================
# 6. Agglomerative Clustering. Clustering gerarchico “bottom-up”: ogni punto parte come cluster singolo e poi vengono uniti passo passo.
# =========================
agglo = AgglomerativeClustering(n_clusters=k)
cluster_df["Agglo"] = agglo.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
for label in range(k):
    subset = cluster_df[cluster_df["Agglo"] == label]
    plt.scatter(subset["Successo Complessivo del Farmaco"], subset["Accordo/Disaccordo degli Utenti"], s=subset["n_reviews"]*5, label=f'Cluster {label}')
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
# 7. Stampa informazioni utili clustering
# =========================
cluster_order = cluster_df.groupby("KPrototypes")["bayes_mean"].mean().sort_values(ascending=False)
cluster_mapping = {num: name for num, name in zip(cluster_order.index, ["Top", "Medio", "Basso"])}
cluster_df["cluster_name"] = cluster_df["KPrototypes"].map(cluster_mapping)

print("\n=== CLUSTERING FARMACI – Condizione:", condition, "===\n")
print("Feature usate per il clustering:")
print(cluster_df[["rating_mean", "rating_std", "n_reviews", "eff_count", "bayes_mean"]].head(), "\n")

print("Cluster assegnati con nomi significativi:")
print(cluster_df[["rating_mean", "bayes_mean", "eff_count", "n_reviews", "cluster_name"]].sort_values("bayes_mean", ascending=False))
