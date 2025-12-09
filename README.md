Ecco un README completo e adatto a un progetto universitario basato sulla tua repo:

---

# Progetto Drug Reviews (Druglib.com)

**Corso:** Principi e modelli della percezione
**Autori:** Anastasiya Kozemko, Mychael Fokou, Camilla Moretti, Gift Aighobahi

---

## Descrizione

Questo progetto analizza il **Drug Review Dataset** di Druglib.com come parte dell’esame del corso “Principi e modelli della percezione”. L’obiettivo è studiare le percezioni degli utenti sui farmaci tramite recensioni, valutazioni, effetti collaterali e condizioni mediche associate, con approcci di **data analysis**, **clustering** e **previsione dei rating**.

---

## Struttura del repository

* `drugLibTrain_final_v4.tsv`, `drugLibTest_final_v4.tsv` — dataset per training e test.
* `pulizia_dataset/` — script per la pulizia e il preprocessing dei dati.
* `analisi_esplorativa/` — script per visualizzazioni e analisi dei dati (distribuzioni, box-plot, heatmap, ecc.).
* `clustering/` — script per applicare algoritmi di clustering sui dati e visualizzare i gruppi.
* `previsione rating/` — script per rielaborazione, normalizzazione e previsione dei rating originali.

---

## Contenuti principali

* Grafici della distribuzione di variabili come rating, efficacia, effetti collaterali.
* Analisi della frequenza delle condizioni mediche tramite bar-plot e lollipop-plot.
* Box-plot per confrontare la distribuzione del rating tra le condizioni più comuni.
* Heatmap delle correlazioni tra variabili numeriche.
* Ranking dei farmaci per condizione usando **Bayesian Rating** per mitigare il bias derivante dal numero differente di recensioni.
* Risultati di clustering con diversi algoritmi per raggruppare farmaci con caratteristiche simili.

---

## Motivazione e obiettivi

* Comprendere come tecniche di **clustering** possano estrarre pattern da dati testuali.
* Studiare rappresentazioni visive di risultati complessi per facilitare l’interpretazione di sentiment, tendenze e strutture latenti.
* Sviluppare competenze pratiche nell’uso di librerie Python per **data science**, **visualizzazione** e **machine learning**.

---

## Dataset

Il dataset originale è disponibile qui: [Drug Review Dataset – UCI](https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com)

---

## Tecnologie e librerie principali

* Python 3.x
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn, Kmodes/KPrototypes
* Altri strumenti di preprocessing e visualizzazione

---

## Istruzioni per l’uso

1. Clonare il repository:

   ```bash
   git clone https://github.com/imNNastya/Progetto-Percezioni.git
   ```
2. Installare le librerie necessarie (es. via `pip install -r requirements.txt`).
3. Pulire e preprocessare i dati con gli script in `pulizia_dataset/`.
4. Eseguire analisi esplorativa (`analisi_esplorativa/`).
5. Applicare clustering (`clustering/`) e valutare i gruppi.
6. Eseguire previsione dei rating (`previsione rating/`).

---
