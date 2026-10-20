# Credit Default Risk Prediction
> Binary classification for loan default risk

This project implements a machine learning pipeline that predicts whether loan applicants will have payment difficulties. The solution supports credit risk assessment using application and loan data.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Modeling Approach](#modeling-approach)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [How to Run](#how-to-run)
- [Architecture Diagram](#architecture-diagram)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)

---

## Problem Statement

The task is a binary classification problem: predict whether a person applying for home credit will be able to repay their debt.

- **Target 1**: Client will have payment difficulties (late payment of more than X days on at least one of the first Y installments of the loan in the sample).
- **Target 0**: Client will repay without such difficulties.

The model must output probabilities that a loan will not be paid for each applicant. The goal is to support lending decisions and default risk management.

---

## Dataset Description

The dataset is based on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/) Kaggle competition. This project uses the primary application files only:

- **`application_train_aai.csv`**: Training data with labels.
- **`application_test_aai.csv`**: Test applications. The file includes a `TARGET` column, which later cells overwrite with predicted class-1 probabilities. No test ROC AUC is computed.

### Dataset characteristics

- **Task**: Binary classification.
- **Features**: Applicant demographics, financial information, employment details, and housing/family information.
- **Class distribution**: Imbalanced (typical for credit risk).
- **Data quality**: Missing values, outliers (e.g. `DAYS_EMPLOYED`), and categorical variables that require encoding.

Data is downloaded automatically when running the notebook’s data-loading section.

---

## Modeling Approach

### Data preprocessing pipeline

1. **Outlier correction**  
   Anomalous values (e.g. `DAYS_EMPLOYED` = 365243) are replaced with NaN.

2. **Categorical encoding**  
   - Binary categoricals: `OrdinalEncoder`.  
   - Multi-category: `OneHotEncoder` (sparse_output=False, handle_unknown='ignore').

3. **Missing value imputation**  
   `SimpleImputer` with median strategy for all numerical features.

4. **Feature scaling**  
   `MinMaxScaler` to scale features to [0, 1].

### Model architecture

The selected model is a tuned **LightGBM classifier** (`lgbm_random`). It has no feature-selection step and no `class_weight`.

- **LGBMClassifier**: n_estimators=200, max_depth=5, learning_rate=0.1, num_leaves=15, subsample=0.6, colsample_bytree=0.8, min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1.

### Model selection

Six models were compared:

| Approach | Validation ROC AUC | Train–val gap |
| --- | ---: | ---: |
| Logistic regression (`C=0.0001`) | 0.6772 | 0.0026 |
| Random forest (default) | 0.7064 | 0.2936 |
| Random forest (tuned) | 0.7302 | 0.0768 |
| LightGBM (`lgbm_random`) | 0.7531 | 0.0425 |
| Logistic pipeline | 0.7470 | 0.0033 |
| LightGBM pipeline | 0.7493 | 0.0417 |

`lgbm_random` has the highest validation ROC AUC (0.7531). That search used `RandomizedSearchCV` (3-fold CV, `scoring='roc_auc'`, 10 iterations, `random_state=42`). The LightGBM pipeline search used 15 iterations and reached 0.7493.

---

## Evaluation Metrics

**Primary metric: ROC AUC (Area Under the ROC Curve)**

ROC AUC is used because it is threshold-independent, suitable for imbalanced data, and reflects probability ranking quality. It is a standard choice for credit risk models.

### Model performance

| Model | Train ROC AUC | Validation ROC AUC | Train–val gap |
| --- | ---: | ---: | ---: |
| Logistic regression (`C=0.0001`) | 0.6798 | 0.6772 | 0.0026 |
| Random forest (default) | 1.0000 | 0.7064 | 0.2936 |
| Random forest (tuned) | 0.8070 | 0.7302 | 0.0768 |
| **LightGBM (`lgbm_random`)** | **0.7956** | **0.7531** | **0.0425** |
| Logistic pipeline | 0.7503 | 0.7470 | 0.0033 |
| LightGBM pipeline | 0.7910 | 0.7493 | 0.0417 |

---

## Results

- **Validation ROC AUC**: 0.7531 for the selected LightGBM model (`lgbm_random`).
- **Generalization**: Train–validation gap of 0.0425.
- `lgbm_random` writes the test-file probabilities. The feature-selection pipeline is not saved.

---

## How to Run

### Prerequisites

- Python 3.8 or higher
- pip
- Git (for cloning)

### Step 1: Clone the repository

```bash
git clone <your-repo-url>
cd credit-default-modeling
```
(Use your actual repo name in place of `credit-default-modeling` if you chose a different one.)

### Step 2: Set up virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

# If you get an execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the notebook

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `credit_default_modeling.ipynb`.
3. Run all cells in order. Data is downloaded in the first section; the notebook covers loading, preprocessing, training, evaluation, and predictions.

### Step 5: Run tests

```bash
pytest tests/
```

### Optional: Code formatting

```bash
isort --profile=black . && black --line-length 88 .
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Raw Data Input                            │
│         (application_train_aai.csv, test files)               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Preprocessing Pipeline                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Outlier      │  │ Categorical  │  │ Missing      │       │
│  │ Correction   │→ │ Encoding     │→ │ Value        │       │
│  │              │  │              │  │ Imputation   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│  ┌──────────────┐                                            │
│  │ Feature      │                                            │
│  │ Scaling      │                                            │
│  │ (MinMax)     │                                            │
│  └──────────────┘                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Training                                  │
│  Selected: LightGBM lgbm_random                              │
│  RandomizedSearchCV, n_iter=10, roc_auc, no class_weight     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Evaluation & Predictions                  │
│  ROC AUC on train/validation; probability outputs           │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
credit-default-modeling/
├── credit_default_modeling.ipynb   # Main notebook
├── GITHUB_UPLOAD.md                # Steps to publish repo with neutral name
├── README.md
├── README_og.md
├── requirements.txt
├── src/
│   ├── config.py
│   ├── data_utils.py
│   └── preprocessing.py
└── tests/
    ├── conftest.py
    ├── test_preprocessing.py
    └── test_data_utils.py
```

---

## Technologies Used

- **Python 3.8+**: Core language
- **Pandas**: Data manipulation
- **NumPy**: Numerical computation
- **Scikit-learn**: Preprocessing, feature selection, evaluation
- **LightGBM**: Gradient boosting classifier
- **Matplotlib / Seaborn**: Visualization
- **Jupyter**: Notebooks
- **Pytest**: Testing
- **Black / isort**: Code formatting

---

## References

- [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk/)
- [ROC and AUC (Google ML Crash Course)](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [LightGBM documentation](https://lightgbm.readthedocs.io/)

---

## License

This project is for educational purposes.

---

## Author

Marissa Singh
