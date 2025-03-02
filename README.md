# 📊 Data Analysis Report On Electricity Demand and Weather Data

## 🌟 Overview
Welcome to the **Data Analysis Report On Electricity Demand and Weather Data** repository! This project contains an in-depth statistical analysis of a dataset, focusing on essential aspects such as missing data handling, summary statistics, outlier analysis, multicollinearity assessment, and model evaluation. Dive in to discover valuable insights! 🚀

## 🔄 Cloning the Repository
To get started with the code, clone this repository to your local machine using the following command:

```bash
git clone https://github.com/nasif1731/DataPreprocessing_and_EDA.git
```

Once cloned, navigate into the directory:

```bash
cd DataPreprocessing_and_EDA
```

## 📂 Project Structure
- raw/: Contains the raw weather and electricity demand data.
- processed/: Directory for saving cleaned and processed data.
- README.md: This file.

## 🧹 Data Preprocessing
Data preprocessing is crucial for ensuring the quality and reliability of the analysis. The following steps were taken:

- 🧼 **Data Cleaning:** Ensured all entries were valid and consistent.
- 📏 **Normalization:** Scaled variables to bring them onto a similar scale.
- 🔢 **Encoding Categorical Variables:** Transformed categorical variables into numerical formats for analysis.

## 📈 Exploratory Data Analysis (EDA)
EDA was conducted to understand the underlying patterns in the dataset. Key visualizations and analyses included:

- 📊 **Histograms:** Showed the distribution of continuous variables.
- 📦 **Boxplots:** Identified outliers and the spread of the data.
- 🔗 **Correlation Matrix:** Assessed relationships between variables.

### Code Snippet for EDA
```python
# Perform Exploratory Data Analysis
def perform_eda(df):
    # Statistical Summary
    print("\nStatistical Summary:\n", df.describe())
    
    # Distribution Analysis
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['electricity_demand'], label='Electricity Demand', color='b')
    plt.xlabel("Timestamp")
    plt.ylabel("Electricity Demand")
    plt.title("Electricity Demand Over Time")
    plt.legend()
    plt.grid()
    plt.show()
```

### 1. 🚫 Handling Missing Data
- **MCAR (Missing Completely at Random):** No correlation between missing values and observed data.
- **MAR (Missing at Random):** Missingness is related to observed but non-missing values.
- **MNAR (Missing Not at Random):** Missingness depends on the missing values themselves.

📝 **Finding:** The dataset contained no missing values, so no imputation was required.

### 2. 📊 Summary Statistics
Each variable's distribution was analyzed for:

- **Central Tendency:** (Mean, Median, Mode)
- **Dispersion:** (Variance, Standard Deviation, IQR)
- **Shape Characteristics:** (Skewness, Kurtosis)

| Variable           | Mean (e)   | Std Dev (e) | Skewness (e) | Kurtosis (e) |
|--------------------|------------|-------------|---------------|---------------|
| Temperature        | 6.60e-17   | 1.00e+00    | 1.80e-01      | -6.07e-01     |
| Electricity Demand  | 2.99e-17   | 1.00e+00    | 2.40e+00      | 5.93e+00      |
| Hour               | 1.17e-16   | 1.00e+00    | -4.64e-04     | -1.20e+00     |
| Day                | 1.26e-17   | 1.00e+00    | 6.67e-03      | -1.19e+00     |
| Month              | -8.17e-17  | 1.00e+00    | -8.23e-03     | -1.21e+00     |
| Year               | 1.04e-13   | 1.00e+00    | 5.07e-03      | -1.48e+00     |
| Day of Week        | 2.63e-17   | 1.00e+00    | 1.63e-03      | -1.25e+00     |
| Is Weekend         | 4.28e-17   | 1.00e+00    | 9.46e-01      | -1.10e+00     |

### 3. 🔍 Outlier Analysis
**Detection Methods**
- 🔢 Z-score Analysis
- 📏 Interquartile Range (IQR) Method
- 📊 Boxplots & Histograms

**Handling Strategy**
- **Electricity Demand:** Extreme values capped to reduce skewness.
- **Other Variables:** No major anomalies found.

### 4. 📉 Multicollinearity Analysis
Variance Inflation Factor (VIF) was used to check for correlation among predictors.

| Variable           | VIF Value |
|--------------------|-----------|
| Temperature        | 1.00      |
| Electricity Demand  | 1.00      |
| Hour               | 1.00      |
| Day                | 1.00      |
| Month              | 1.00      |
| Year               | 1.00      |
| Day of Week        | 1.00      |
| Is Weekend         | 1.00      |

📝 **Conclusion:** No multicollinearity detected, ensuring model interpretability.

### 5. 📊 Model Performance
- **Stationarity Analysis (Augmented Dickey-Fuller Test)**
  - ADF Statistic: -24.36
  - p-value: 0.00
  - Inference: Data is stationary, suitable for time series modeling.

### Code Snippet for Model Training
```python
def train_regression(df):
    features = ['hour', 'day_of_week', 'month', 'temperature', 'year', 'is_weekend']
    X = df[features]
    y = df['electricity_demand']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
```

**Regression Model Metrics**

| Metric | Value  |
|--------|--------|
| MSE    | 0.155  |
| RMSE   | 0.394  |
| R²     | 0.003  |

📝 **Findings:**
- Low R² suggests limited predictive power.
- Further feature engineering or alternative modeling can improve results.

## 🚀 Conclusion
- Dataset is clean and standardized for analysis.
- No missing values or multicollinearity issues.
- Electricity demand distribution improved after outlier treatment.

## 🛠️ Requirements
Ensure you have the following packages installed:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels
- scikit-learn
You can install the required packages using:
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
```

🔮 **Future Work:** Explore advanced feature engineering or alternative models.

This repository provides a structured, reproducible analysis pipeline. Happy analyzing! 🎉
