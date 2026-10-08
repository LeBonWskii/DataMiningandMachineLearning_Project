# Data Mining and Machine Learning: Hotel Booking Cancellation Prediction

Progetto di Data Mining e Machine Learning dedicato alla **previsione delle cancellazioni delle prenotazioni alberghiere**. Il lavoro comprende l'analisi esplorativa dei dati, la costruzione di una pipeline di preprocessing e feature engineering, il confronto di diversi classificatori, l'interpretazione delle predizioni e un prototipo interattivo realizzato con Streamlit.

## Scopo del progetto

L'obiettivo è stimare se una prenotazione verrà cancellata utilizzando informazioni sulle sue caratteristiche: anticipo della prenotazione, durata del soggiorno, composizione degli ospiti, canale di distribuzione, deposito, storico del cliente e tariffa.

Il problema è formulato come una **classificazione binaria**, con la variabile target `IsCanceled`:

- **0**: prenotazione non cancellata.
- **1**: prenotazione cancellata.

La finalità applicativa è supportare la gestione alberghiera nell'identificazione delle prenotazioni a rischio e nella pianificazione dell'occupazione. Il progetto studia sia le prestazioni predittive sia i fattori che contribuiscono alle decisioni del modello.

## Dati utilizzati

I dati sono contenuti in due file nella cartella [Dataset](Dataset):

| File | Tipologia di struttura | Prenotazioni |
| --- | --- | ---: |
| [H1.csv](Dataset/H1.csv) | Resort hotel | 40.060 |
| [H2.csv](Dataset/H2.csv) | City hotel | 79.330 |
| **Totale** | | **119.390** |

I due dataset vengono uniti aggiungendo la variabile `HotelType`. Nell'insieme originale sono presenti 75.166 prenotazioni non cancellate e 44.224 cancellate: questo sbilanciamento viene considerato nella fase di training.

## Lavoro realizzato

### 1. Analisi esplorativa dei dati

Il notebook [1. EDA.ipynb](notebooks/1.%20EDA.ipynb) analizza:

- Distribuzione delle cancellazioni per tipologia di hotel, anno e mese.
- Correlazioni tra variabili numeriche e associazioni tra variabili categoriche e target tramite test chi-quadrato.
- Distribuzioni e valori anomali delle variabili numeriche.
- Relazione tra cancellazioni, anticipo della prenotazione (`LeadTime`) e stagionalità.
- Andamento della tariffa media giornaliera e intervallo tra cancellazione e arrivo previsto.

### 2. Preprocessing e feature engineering

Il notebook [2. Preprocessing.ipynb](notebooks/2.%20Preprocessing.ipynb) prepara i dati attraverso la gestione dei valori mancanti e delle categorie `NULL` e `Undefined`, la rimozione delle prenotazioni senza ospiti e il raggruppamento delle categorie poco frequenti di `Agent` e `Company` con una soglia del 2%.

Vengono inoltre escluse variabili individuate nell'analisi come potenziali fonti di **data leakage**, tra cui `Country`, `AssignedRoomType`, `ReservationStatus` e `ReservationStatusDate`.

Il transformer personalizzato [ADRThirdQuartileDeviationTransformer](notebooks/utils/FeatureTransformer.py) costruisce la feature:

```text
ADRThirdQuartileDeviation = ADR / terzo quartile dell'ADR del gruppo
```

I gruppi sono definiti da canale di distribuzione, tipologia di camera prenotata, anno e settimana di arrivo. I quartili vengono appresi sui dati passati al metodo `fit`. Dopo la trasformazione vengono eliminate la tariffa originale, le variabili di arrivo e `ReservedRoomType`.

Il [preprocessor](notebooks/utils/preprocessor.py) applica `StandardScaler` alle feature numeriche e `OneHotEncoder` a quelle categoriche.

### 3. Training, ottimizzazione e valutazione

Il notebook [3. Training.ipynb](notebooks/3.%20Training.ipynb) confronta **otto classificatori**:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Random Forest
- AdaBoost
- XGBoost
- LightGBM
- CatBoost

La procedura comprende:

- Suddivisione train/test circa **75%/25%**, effettuata separatamente in ciascun blocco anno-mese e stratificata per target.
- Cross-validation stratificata a **5 fold**.
- Confronto delle pipeline con e senza oversampling **SMOTENC**.
- Ricerca degli iperparametri tramite **GridSearchCV**, usando l'F1 score come criterio.
- Confronto statistico degli F1 score sui fold con test di **Friedman** e post-hoc di **Nemenyi**.
- Valutazione di cinque modelli sul test set tramite accuracy, precision, recall, F1 score, ROC AUC, curve ROC e matrici di confusione.

Lo split include prenotazioni degli stessi mesi sia nel training sia nel test; i risultati descrivono quindi questo protocollo di valutazione.

### 4. Interpretabilità e applicazione interattiva

Il notebook [4. Explainability.ipynb](notebooks/4.%20Explainability.ipynb) approfondisce il comportamento di Random Forest mediante:

- Feature importance del modello.
- Analisi globale con valori **SHAP**, grafici a barre e beeswarm.
- Spiegazioni locali con waterfall plot per predizioni corrette, errate, vicine alla soglia decisionale e con contributi SHAP elevati.

L'app [Streamlit](notebooks/app.py) consente di caricare un CSV di prenotazioni e visualizzare la classe prevista, la probabilità di cancellazione e una fascia di rischio. Presenta inoltre indicatori riepilogativi e un istogramma delle probabilità.

## Risultati

Le metriche riportate nel file [holdout_results.csv](notebooks/holdout_results.csv) sono:

| Modello | Accuracy | Precision | Recall | F1 score | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Random Forest** | **0,844** | **0,794** | **0,781** | **0,788** | **0,914** |
| LightGBM | 0,835 | 0,792 | 0,752 | 0,772 | 0,902 |
| XGBoost | 0,831 | 0,784 | 0,751 | 0,767 | 0,900 |
| CatBoost | 0,827 | 0,785 | 0,736 | 0,759 | 0,892 |
| KNN | 0,828 | 0,768 | 0,769 | 0,768 | 0,894 |

Precision, recall e F1 score si riferiscono alla classe positiva, cioè alle prenotazioni cancellate. I valori provengono dagli esperimenti salvati nella repository.

**Random Forest ottiene i valori più alti in tutte le metriche della tabella** ed è utilizzato per l'analisi di interpretabilità e per l'esportazione della pipeline destinata all'app.

## Esplorare ed eseguire il progetto

I notebook vanno aperti dalla cartella `notebooks` ed eseguiti nell'ordine **EDA → Preprocessing → Training → Explainability**.

Le principali tecnologie utilizzate sono Python, Jupyter, pandas, NumPy, Matplotlib, seaborn, scikit-learn, imbalanced-learn, feature-engine, XGBoost, LightGBM, CatBoost, SciPy, scikit-posthocs, SHAP, joblib e Streamlit.

Per l'esecuzione:

- I notebook EDA e Preprocessing leggono `../Dataset/h1.csv` e `../Dataset/h2.csv`, mentre i file nella repository sono `H1.csv` e `H2.csv`. Su sistemi che distinguono maiuscole e minuscole occorre adeguare i percorsi nelle celle di caricamento.
- Preprocessing genera `Dataset/df_cleaned.csv`, necessario ai notebook successivi.
- L'ultima cella di Training genera `notebooks/hotel_pipeline.pkl`, necessario all'app. La pipeline esportata comprende feature engineering, preprocessing e Random Forest; in questa fase viene addestrata sull'intero dataset pulito senza SMOTENC.
- Il dataset pulito e la pipeline serializzata sono output da generare localmente e non sono inclusi nella repository. Non è presente un file delle dipendenze con versioni fissate.

Dopo aver generato la pipeline, avviare l'app dalla cartella `notebooks`:

```bash
cd notebooks
streamlit run app.py
```

Il CSV caricato nell'app deve contenere le feature richieste dalla pipeline, con nomi e categorie coerenti con i dati di training, inclusa `HotelType`.

## Documentazione e output

| Risorsa | Contenuto |
| --- | --- |
| [documentation_Falaschi.pdf](documentation_Falaschi.pdf) | Relazione del progetto |
| [presentation_Falaschi.pdf](presentation_Falaschi.pdf) | Presentazione del progetto |
| [default_training.csv](notebooks/default_training.csv) | Metriche dei modelli con parametri di default |
| [grid_search_results.csv](notebooks/grid_search_results.csv) | Risultati della ricerca degli iperparametri |
| [best_model_versions.csv](notebooks/best_model_versions.csv) | Scelta tra versioni di default e ottimizzate |
| [holdout_results.csv](notebooks/holdout_results.csv) | Metriche del confronto sul test set riportato sopra |

## Licenza

La repository è distribuita con [licenza MIT](LICENSE).
