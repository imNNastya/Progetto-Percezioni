# Progetto Drug reviews (Druglib.com) 

**Corso:** Principi e modelli della percezione  
**Autori:** Anastasiya Kozemko, Mychael Fokou, Camilla Moretti, Gift Aighobahi  

---

## 🔎 Descrizione
 
Questo progetto si occupa dell’analisi di un dataset reale — il Drug Review Dataset — come parte dell’esame del corso “Principi e modelli della percezione”. L’obiettivo è analizzare le percezioni degli utenti (recensioni) sui farmaci: valutazioni, effetti collaterali, condizioni mediche associate, ecc. 

---

## 📂 Struttura del repo

- `drugLibTrain_final_v4.tsv`, `drugLibTest_final_v4.tsv` — dataset utilizzato per training e test
- `analisi_esplorativa/` - cartella che contiene script per generare grafici/plot che aiutano a interpretare i risultati 
- `clustering/` — cartella che contiene script per eseguire l’algoritmo di clustering sui dati con le sue visualizzazioni
- `previsione rating/` — cartella che contiene la logica per la parte di rating (rielaborazione / previsione / normalizzazione dei rating originali)  
- `pulizia_dataset/` - cartella che contiene la logica per la pulizia e pre processing del dataset
- ... [ finisci a fine riunione )
---
## 📊 Cosa troverai  
- Grafici che mostrano la distribuzione di variabili come rating, efficacia, effetti collaterali. 
- Analisi della frequenza delle condizioni mediche (“condition_standardized”) tramite bar-plot / lollipop-plot. 
- Box-plot per confrontare la distribuzione del rating tra le top condizioni mediche più comuni. 
- Heatmap delle correlazioni tra variabili numeriche. 
- Ranking di farmaci per condizione (usando Bayesian Rating) per mitigare bias da differente numero di recensioni. 
- Risultati di clustering (diversi algoritmi) per raggruppare farmaci 
- PERCHE LA PARTE DI MIKE NON LA PRENDE 
---

## 💡 Motivazione e obiettivi

* Comprendere come tecniche di clustering possano aiutare a estrarre pattern da dati testuali (recensioni, opinioni, feedback).
* Studiare come rappresentare visivamente risultati complessi per facilitare l’interpretazione e l’analisi di sentiment, tendenze e strutture latenti nei dati.
* Sviluppare competenze pratiche nell’utilizzo di librerie Python per data science e visualizzazione, consolidando concetti affrontati nel corso “Principi e modelli della percezione”.

---
```
