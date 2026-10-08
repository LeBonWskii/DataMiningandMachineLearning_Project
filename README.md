# Data Mining and Machine Learning: Hotel Booking Cancellation Prediction

A Data Mining and Machine Learning project focused on **predicting hotel booking cancellations**. The work covers exploratory data analysis, a preprocessing and feature engineering pipeline, comparison of multiple classifiers, prediction explainability, and an interactive prototype built with Streamlit.

## Project objective

The goal is to estimate whether a booking will be canceled using its characteristics, including booking lead time, length of stay, guest composition, distribution channel, deposit type, customer history, and room rate.

The task is formulated as **binary classification**, with `IsCanceled` as the target variable:

- **0**: booking not canceled.
- **1**: booking canceled.

The intended application is to support hotel management in identifying bookings at risk of cancellation and planning occupancy. The project examines both predictive performance and the factors contributing to model decisions.

## Dataset

The data is stored in two files in the [Dataset](Dataset) directory:

| File | Hotel type | Bookings |
| --- | --- | ---: |
| [H1.csv](Dataset/H1.csv) | Resort hotel | 40,060 |
| [H2.csv](Dataset/H2.csv) | City hotel | 79,330 |
| **Total** | | **119,390** |

The datasets are combined by adding the `HotelType` variable. The original combined dataset contains 75,166 non-canceled bookings and 44,224 canceled bookings; this class imbalance is addressed during model training.

## Work completed

### 1. Exploratory data analysis

The [1. EDA.ipynb](notebooks/1.%20EDA.ipynb) notebook examines:

- Cancellation distributions by hotel type, year, and month.
- Correlations between numerical variables and associations between categorical variables and the target using chi-square tests.
- Distributions and outliers in numerical variables.
- Relationships between cancellations, booking lead time (`LeadTime`), and seasonality.
- Average daily rate trends and the interval between cancellation and scheduled arrival.

### 2. Preprocessing and feature engineering

The [2. Preprocessing.ipynb](notebooks/2.%20Preprocessing.ipynb) notebook prepares the data by handling missing values and `NULL` and `Undefined` categories, removing bookings with no guests, and grouping infrequent categories in `Agent` and `Company` using a 2% threshold.

Variables identified during the analysis as potential sources of **data leakage** are also excluded, including `Country`, `AssignedRoomType`, `ReservationStatus`, and `ReservationStatusDate`.

The custom [ADRThirdQuartileDeviationTransformer](notebooks/utils/FeatureTransformer.py) creates the following feature:

```text
ADRThirdQuartileDeviation = ADR / third quartile of ADR within the group
```

Groups are defined by distribution channel, reserved room type, arrival year, and arrival week. Quartiles are learned from the data supplied to the `fit` method. After transformation, the original rate, arrival variables, and `ReservedRoomType` are removed.

The [preprocessor](notebooks/utils/preprocessor.py) applies `StandardScaler` to numerical features and `OneHotEncoder` to categorical features.

### 3. Model training, optimization, and evaluation

The [3. Training.ipynb](notebooks/3.%20Training.ipynb) notebook compares **eight classifiers**:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Random Forest
- AdaBoost
- XGBoost
- LightGBM
- CatBoost

The workflow includes:

- An approximately **75%/25% train/test split**, performed separately within each year-month block and stratified by the target.
- **5-fold** stratified cross-validation.
- Comparison of pipelines with and without **SMOTENC** oversampling.
- Hyperparameter tuning with **GridSearchCV**, using F1 score as the optimization criterion.
- Statistical comparison of fold-level F1 scores using the **Friedman** test and **Nemenyi** post-hoc test.
- Evaluation of five models on the test set using accuracy, precision, recall, F1 score, ROC AUC, ROC curves, and confusion matrices.

The split includes bookings from the same months in both training and test sets; the reported results therefore reflect this evaluation protocol.

### 4. Explainability and interactive application

The [4. Explainability.ipynb](notebooks/4.%20Explainability.ipynb) notebook investigates Random Forest behavior through:

- Model feature importance.
- Global analysis with **SHAP** values, bar charts, and beeswarm plots.
- Local explanations with waterfall plots for correct predictions, misclassifications, predictions close to the decision threshold, and samples with large SHAP contributions.

The [Streamlit application](notebooks/app.py) lets users upload a bookings CSV and view the predicted class, cancellation probability, and risk category. It also displays summary indicators and a histogram of predicted probabilities.

## Results

The metrics recorded in [holdout_results.csv](notebooks/holdout_results.csv) are:

| Model | Accuracy | Precision | Recall | F1 score | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Random Forest** | **0.844** | **0.794** | **0.781** | **0.788** | **0.914** |
| LightGBM | 0.835 | 0.792 | 0.752 | 0.772 | 0.902 |
| XGBoost | 0.831 | 0.784 | 0.751 | 0.767 | 0.900 |
| CatBoost | 0.827 | 0.785 | 0.736 | 0.759 | 0.892 |
| KNN | 0.828 | 0.768 | 0.769 | 0.768 | 0.894 |

Precision, recall, and F1 score refer to the positive class, i.e., canceled bookings. These values come from the experiments saved in the repository.

**Random Forest achieves the highest values across all metrics in the table** and is used for the explainability analysis and for exporting the pipeline consumed by the application.

## Exploring and running the project

Open the notebooks from the `notebooks` directory and run them in the order **EDA → Preprocessing → Training → Explainability**.

The main technologies used are Python, Jupyter, pandas, NumPy, Matplotlib, seaborn, scikit-learn, imbalanced-learn, feature-engine, XGBoost, LightGBM, CatBoost, SciPy, scikit-posthocs, SHAP, joblib, and Streamlit.

Execution notes:

- The EDA and Preprocessing notebooks read `../Dataset/h1.csv` and `../Dataset/h2.csv`, while the repository files are named `H1.csv` and `H2.csv`. On case-sensitive systems, adjust the paths in the data-loading cells.
- Preprocessing generates `Dataset/df_cleaned.csv`, which is required by the subsequent notebooks.
- The last cell in Training generates `notebooks/hotel_pipeline.pkl`, which is required by the application. The exported pipeline includes feature engineering, preprocessing, and Random Forest; at this stage, it is fitted on the entire cleaned dataset without SMOTENC.
- The cleaned dataset and serialized pipeline are outputs to generate locally and are not included in the repository. No dependency file with pinned versions is provided.

After generating the pipeline, launch the application from the `notebooks` directory:

```bash
cd notebooks
streamlit run app.py
```

The CSV uploaded to the application must contain the features required by the pipeline, with column names and categories consistent with the training data, including `HotelType`.

## Documentation and outputs

| Resource | Contents |
| --- | --- |
| [documentation_Falaschi.pdf](documentation_Falaschi.pdf) | Project report |
| [presentation_Falaschi.pdf](presentation_Falaschi.pdf) | Project presentation |
| [default_training.csv](notebooks/default_training.csv) | Metrics for models with default parameters |
| [grid_search_results.csv](notebooks/grid_search_results.csv) | Hyperparameter search results |
| [best_model_versions.csv](notebooks/best_model_versions.csv) | Selection between default and optimized model versions |
| [holdout_results.csv](notebooks/holdout_results.csv) | Test-set comparison metrics reported above |

## License

This repository is distributed under the [MIT License](LICENSE).
