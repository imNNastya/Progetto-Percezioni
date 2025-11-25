import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Caricamento train + test
train = pd.read_csv("drugLibTrain_raw.tsv", sep="\t")
test = pd.read_csv("drugLibTest_raw.tsv", sep="\t")
df = pd.concat([train, test], ignore_index=True)

# 2. Seleziona condizione
condition = "anxiety"
df_cond = df[df["condition"] == condition]

print(f"Recensioni trovate per '{condition}': {len(df_cond)}")

# 3. Consideriamo tutti i farmaci
df_valid = df_cond

# 4. Calcolo media rating
rating_column = "rating"
best_drugs = df_valid.groupby("urlDrugName")[rating_column].mean().sort_values(ascending=False)

# TOP 10
top = best_drugs.head(10)
top_drugs = top.index.tolist()
print("\nTop 10 farmaci:")
print(top)

# Colori coerenti
main_color = "#1f77b4"
desaturated = "#aec7e8"

# (1) DOT PLOT / LOLLIPOP
#plt.figure(figsize=(10, 6))
#plt.hlines(y=top.index, xmin=0, xmax=top.values, color=desaturated, linewidth=4)
#plt.scatter(top.values, top.index, color=main_color, s=120)
#plt.title(f"Dot/Lollipop Plot – {condition}", fontsize=14)
#plt.xlabel("Rating medio")
#plt.gca().invert_yaxis()
#plt.tight_layout()
#plt.show()

plt.figure(figsize=(10, 6))

winner = top.index[0]  # farmaco migliore

for i, drug in enumerate(top.index):
    value = top[drug]
    color = main_color if drug == winner else desaturated
    lw = 4 if drug == winner else 2
    size = 600 if drug == winner else 400
    
    # linea lollipop
    plt.hlines(y=drug, xmin=0, xmax=value, color=color, linewidth=lw, alpha=0.8)
    # pallino finale
    plt.scatter(value, drug, color=color, s=size, edgecolor='white', linewidth=1.5, zorder=5)
    # rating **dentro il pallino**
    plt.text(value, i, f"{value:.1f}", va='center', ha='center', fontsize=7, color='white', fontweight='bold', zorder=6)

plt.title(f"Lollipop Plot Miglioramento Visivo – {condition}", fontsize=16)
plt.xlabel("Rating medio")
plt.ylabel("Farmaco")
plt.xlim(0, top.max() + 2)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# (2) BOXPLOT
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_valid[df_valid["urlDrugName"].isin(top_drugs)],
    x="rating",
    y="urlDrugName",
    palette="Blues"
)
plt.title(f"Distribuzione dei rating – Top 10 farmaci per {condition}", fontsize=14)
plt.xlabel("Rating")
plt.ylabel("Farmaco")
plt.tight_layout()
plt.show()

# (3) HEATMAP EFFECTIVENESS
# Convert effectiveness in numerico
effect_map = {"Ineffective": 1,"Marginally Effective":2, "Moderately Effective": 3, "Considerably Effective": 4, "Highly Effective":5}
df_valid["effectiveness_num"] = df_valid["effectiveness"].map(effect_map)

# Calcolo media effectiveness per farmaco
eff_table = (
    df_valid[df_valid["urlDrugName"].isin(top_drugs)]
    .groupby("urlDrugName")["effectiveness_num"]
    .mean()
    .to_frame()
)

plt.figure(figsize=(10, 4))
sns.heatmap(
    eff_table.sort_values("effectiveness_num", ascending=False),
    annot=True,
    cmap="Blues",
    cbar=True
)
plt.title(f"Heatmap – Effectiveness media Top 10 farmaci ({condition})", fontsize=14)
plt.tight_layout()
plt.show()

# Violin plot
# Top farmaco
winner = top_drugs[0]

# Creiamo una colonna per il colore
def violin_color(drug):
    if drug == winner:
        return "#0055ff"  # blu super saturo per il vincitore
    else:
        return "#aec7e8"  # blu chiaro desaturato

colors = df_valid[df_valid["urlDrugName"].isin(top_drugs)]["urlDrugName"].map(violin_color)

plt.figure(figsize=(10,6))
sns.violinplot(
    data=df_valid[df_valid["urlDrugName"].isin(top_drugs)],
    x="rating",
    y="urlDrugName",
    palette=colors.tolist(),
    inner="box",
    scale="width"
)

# Pallino per il rating medio del vincitore
mean_rating = top[winner]
plt.scatter(mean_rating, winner, color="#ff0000", s=120, zorder=10, label="Miglior farmaco")

plt.title(f"Violin Plot – Distribuzione rating Top 10 farmaci ({condition})", fontsize=14)
plt.xlabel("Rating")
plt.ylabel("Farmaco")
plt.xlim(1, 10)
plt.legend()
plt.tight_layout()
plt.show()

# Cosa fa questo codice:
# estrae i farmaci della condition scelta
# crea le feature necessarie
# normalizza i dati
# esegue K-Means
# usa PCA per visualizzare i cluster in 2D

# (4) CLUSTERING K-MEANS – Solo per la condition scelta
print("\n=== CLUSTERING FARMACI PER CONDITION ===")

# ---- 1) PREPARAZIONE DATI PER IL CLUSTERING ----
cluster_df = (
    df_valid
    .groupby("urlDrugName")
    .agg({
        "rating": ["mean", "std", "count"],
        "effectiveness_num": "mean"
    })
)

cluster_df.columns = ["rating_mean", "rating_std", "n_reviews", "eff_mean"]
cluster_df = cluster_df.fillna(0)

print("\nFeature usate per il clustering:")
print(cluster_df.head())

# ---- 2) SCALING ----
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_df)

# ---- 3) K-MEANS ----
k = 3  # numero di cluster
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)
cluster_df["cluster_num"] = cluster_labels

# ---- 4) ASSEGNA NOMI SIGNIFICATIVI AI CLUSTER ----
# Ordiniamo i cluster per rating medio
cluster_order = cluster_df.groupby("cluster_num")["rating_mean"].mean().sort_values(ascending=False)
cluster_mapping = {num: name for num, name in zip(cluster_order.index, ["Top", "Medio", "Basso"])}
# Top - farmaci con rating/efficacy più alti (secondo il k-means)
# Medio - farmaci con valori medi
# Basso - farmaci con valori più bassi
cluster_df["cluster_name"] = cluster_df["cluster_num"].map(cluster_mapping)

print("\nCluster assegnati con nomi significativi:")
print(cluster_df[["rating_mean","eff_mean","n_reviews","cluster_name"]])

# ---- 5) PCA PER VISUALIZZAZIONE 2D ----
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(scaled_features)
cluster_df["PC1"] = pca_coords[:, 0]
cluster_df["PC2"] = pca_coords[:, 1]

# ---- 6) GRAFICO DEI CLUSTER ----
plt.figure(figsize=(10, 7))

# palette
palette_dict = {"Top": "#1f77b4", "Medio": "#ff7f0e", "Basso": "#2ca02c"}

for cluster in cluster_df["cluster_name"].unique():
    subset = cluster_df[cluster_df["cluster_name"] == cluster]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        s=subset["n_reviews"] * 5,
        label=f"{cluster} (n={len(subset)})",
        alpha=0.7,
        color=palette_dict[cluster]
    )

# Etichette dei farmaci più rilevanti
for drug, row in cluster_df.iterrows():
    if row["n_reviews"] > 5:
        plt.text(row["PC1"], row["PC2"], drug, fontsize=8)

plt.title(f"Clustering K-Means dei farmaci – Condition: {condition}", fontsize=16)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()