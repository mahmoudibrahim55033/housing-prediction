# 🏠 Housing Price Prediction

A machine learning project that predicts residential housing prices using a variety of regression models and ensemble techniques. The project covers the full ML pipeline — from exploratory data analysis and data cleaning to model comparison and stacking.

---

## 📁 Dataset

- **File:** `Housing.csv`
- **Rows:** 545 records
- **Features:** 13 columns including structural, locational, and amenity attributes

| Feature | Description |
|---|---|
| `price` | Target variable — sale price of the house |
| `area` | Total area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of floors |
| `parking` | Number of parking spaces |
| `mainroad` | Whether the house faces the main road (yes/no) |
| `guestroom` | Availability of a guest room (yes/no) |
| `basement` | Presence of a basement (yes/no) |
| `hotwaterheating` | Hot water heating system (yes/no) |
| `airconditioning` | Air conditioning available (yes/no) |
| `prefarea` | Located in a preferred area (yes/no) |
| `furnishingstatus` | Furnished / semi-furnished / unfurnished |

---

## 🔍 Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Distribution histograms for key numerical features
- Box plots for outlier detection
- Missing value and duplicate checks

### 2. Data Cleaning
- IQR-based outlier removal for `price` and `area`

### 3. Preprocessing
- **Encoding:** One-hot encoding for categorical (yes/no) features using `pd.get_dummies`
- **Scaling:** `StandardScaler` applied to numeric features via `ColumnTransformer`
- **Target Transformation:** Log transformation (`np.log1p`) on `price` to normalize skewed distribution
- **Train/Test Split:** 80/20 split with `random_state=42`

---

## 🤖 Models Trained

| Model | Notes |
|---|---|
| Linear Regression | Baseline model |
| Polynomial Regression (degree=2) | Captures non-linear relationships |
| SVR (Support Vector Regression) | RBF kernel |
| KNN Regression | Tuned with GridSearchCV (k from 1–29) |
| Lasso Regression | L1 regularization (α=0.001) |
| Ridge Regression | L2 regularization (α=0.001), cross-validated |

### Ensemble Methods

| Model | Configuration |
|---|---|
| Bagging | 100 Decision Trees (max_depth=3) |
| AdaBoost | 50 estimators |
| Gradient Boosting | 50 estimators, learning_rate=0.1, max_depth=3 |
| XGBoost | colsample_bytree=0.3, max_depth=5, alpha=10 |
| Stacking | Random Forest + Gradient Boosting → Linear Regression meta-learner |

---

## 📊 Evaluation Metrics

Models were evaluated using:
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination
- **5-Fold Cross Validation** (on Linear Regression and Ridge)

> All metrics computed on log-transformed target values.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/mahmoudibrahim55033/housing-prediction.git
   cd housing-prediction
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook Housing-pred.ipynb
   ```

> Make sure `Housing.csv` is in the same directory as the notebook.

---

## 👤 Author

**mahmoudibrahim55033** — [GitHub Profile](https://github.com/mahmoudibrahim55033)
