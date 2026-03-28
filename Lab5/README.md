# 📊 Evidently AI – Data Drift Analysis on UCI Bank Marketing Dataset

This project demonstrates the use of **Evidently AI** to detect and visualize data drift between two temporal subsets of the **UCI Bank Marketing dataset**.

Data drift analysis helps monitor the stability of data distributions over time, ensuring the reliability of machine learning models in production. This lab simulates a realistic production scenario where a model trained on peak-season data encounters off-season distributional shifts.

---

## 🧠 Overview

- The experiment evaluates **dataset drift** between two groups of clients based on the **month of contact**.
- The **reference dataset** corresponds to clients contacted in **May** (peak campaign activity).
- The **production dataset** corresponds to clients contacted in **November** (low-activity off-season).
- Evidently AI computes drift metrics for each feature, while **custom statistical tests** (PSI, Jensen-Shannon Divergence, KS-test, Chi-Square) provide additional interpretability.

---

## 📦 Tools and Libraries Used

- 🧮 Python 3
- 📈 Evidently AI (v0.7.0+)
- 🐼 Pandas, NumPy
- 🔢 scikit-learn, SciPy
- 🎨 Matplotlib, Seaborn
- 📚 ucimlrepo (for dataset retrieval)

---

## 📂 Dataset

The **Bank Marketing dataset** ([UCI ID: 222](https://archive.ics.uci.edu/dataset/222/bank+marketing)) contains **45,211 records** from a Portuguese bank's phone-based direct marketing campaign. Each record represents one client contact, with attributes covering demographics, financial status, campaign details, and outcome.

### Key Features

| Feature    | Type        | Description                                           |
|------------|-------------|-------------------------------------------------------|
| age        | Numerical   | Age of the client                                     |
| job        | Categorical | Type of job (e.g., management, technician, blue-collar) |
| marital    | Categorical | Marital status (married, single, divorced)            |
| education  | Categorical | Education level (primary, secondary, tertiary)        |
| default    | Categorical | Whether the client has credit in default (yes/no)     |
| balance    | Numerical   | Average yearly balance in euros                       |
| housing    | Categorical | Whether the client has a housing loan (yes/no)        |
| loan       | Categorical | Whether the client has a personal loan (yes/no)       |
| contact    | Categorical | Communication type (cellular, telephone, unknown)     |
| duration   | Numerical   | Duration of last contact in seconds                   |
| campaign   | Numerical   | Number of contacts during this campaign               |
| pdays      | Numerical   | Days since last contact from a previous campaign (-1 = not contacted) |
| previous   | Numerical   | Number of contacts before this campaign               |
| poutcome   | Categorical | Outcome of the previous campaign (success, failure, other, unknown) |
| y          | Categorical | Target — did the client subscribe to a term deposit? (yes/no) |

### Data Split Strategy

| Subset     | Filter         | Records | Description                          |
|------------|----------------|---------|--------------------------------------|
| Reference  | `month == may` | 13,766  | Peak campaign month (high activity)  |
| Production | `month == nov` | 3,970   | Off-season month (low activity)      |

---

## ⚙️ Steps and Methodology

1. **Dataset Loading**: Fetched using `ucimlrepo.fetch_ucirepo(id=222)` and combined into a single DataFrame.
2. **Schema Definition**: Columns divided into 6 numerical and 9 categorical types using Evidently's `DataDefinition()`.
3. **Data Splitting**: Split by `month` column — May (reference) vs November (production).
4. **Evidently Drift Detection**: `DataDriftPreset()` generates an interactive HTML dashboard with per-feature drift analysis.
5. **Manual Drift Detection**: Independent KS-test (numerical) and Chi-Square test (categorical) for validation.
6. **Additional Statistical Tests**: PSI and Jensen-Shannon Divergence computed from scratch for numerical features.
7. **Feature Importance Ranking**: Composite score combining drift test statistic and PSI to rank features by drift severity.
8. **Visualization**: Histograms, box plots, categorical bar charts, and PSI severity bar chart.

---

## 📊 Results

| Metric                  | Value         |
|-------------------------|---------------|
| Total Columns Analyzed  | 15            |
| Drifted Columns         | 14            |
| Drift Share             | 93.33%        |
| Dataset Drift Detected  | ✅ YES        |
| Threshold               | 0.5           |

### Top Drifted Features (by Composite Score)

| Rank | Feature   | Test Statistic | PSI    | Composite Score |
|------|-----------|----------------|--------|-----------------|
| 1    | housing   | 2007.92        | 0.0    | 0.5000          |
| 2    | pdays     | 0.1327         | 0.9764 | 0.5000          |
| 3    | job       | 854.13         | 0.0    | 0.2127          |
| 4    | education | 427.12         | 0.0    | 0.1063          |
| 5    | age       | 0.1315         | 0.1476 | 0.0756          |

### PSI Analysis (Numerical Features)

| Feature  | PSI    | Verdict      |
|----------|--------|--------------|
| pdays    | 0.9764 | 🔴 Major     |
| age      | 0.1476 | 🟡 Moderate  |
| balance  | 0.1094 | 🟡 Moderate  |
| campaign | 0.0974 | 🟢 Stable    |
| previous | 0.0084 | 🟢 Stable    |
| duration | 0.0075 | 🟢 Stable    |

---

## 📈 Example Code

```python
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

data_definition = DataDefinition(
    numerical_columns=numerical_features,
    categorical_columns=categorical_features
)

ref_dataset = Dataset.from_pandas(ref_data, data_definition=data_definition)
prod_dataset = Dataset.from_pandas(prod_data, data_definition=data_definition)

report = Report([DataDriftPreset()])
my_eval = report.run(prod_dataset, ref_dataset)
my_eval.save_html("bank_marketing_drift_report.html")
```

---

## 💡 Key Insights

### 🧮 Statistical Summary
The analysis identified drift in **14 out of 15 features (93.33%)**, far exceeding the 0.5 dataset drift threshold. This confirms substantial distributional change between peak-season (May) and off-season (November) client contacts.

### ⚖️ Feature-Level Drift Highlights
- **housing** showed the strongest categorical drift (Chi-Square = 2007.92), indicating a major shift in housing loan distribution across seasons.
- **pdays** had the highest PSI (0.9764 — Major), reflecting significant changes in how recently clients were previously contacted.
- **job** and **education** showed strong categorical drift, suggesting demographic composition differences between campaign periods.
- **default** was the only feature that did NOT drift (p = 0.204), indicating credit default status remained consistent.

### 📈 Visual Analysis
- Histograms revealed clear distribution shifts in `balance` and `age` between May and November.
- Box plots showed differences in score spreads, particularly for `pdays` and `balance`.
- Categorical bar charts highlighted proportional shifts in `job` types and `education` levels across the two periods.
- PSI bar chart provided an at-a-glance severity view with threshold lines.

### 📐 Statistical Tests Applied
- **Kolmogorov-Smirnov (KS) test**: For numerical features — compares cumulative distributions.
- **Chi-Square test**: For categorical features — tests independence of distributions.
- **Population Stability Index (PSI)**: Measures magnitude of distribution shift (0.10/0.25 thresholds).
- **Jensen-Shannon Divergence (JSD)**: Symmetric, bounded divergence metric (0 = identical, 1 = maximum divergence).

---

## 📁 Generated Output Files

| File                              | Description                                    |
|-----------------------------------|------------------------------------------------|
| `bank_marketing_drift_report.html`| Evidently interactive drift dashboard           |
| `drift_summary_table.csv`        | Exportable per-feature drift results            |
| `numerical_histograms.png`       | Overlapping density histograms                  |
| `numerical_boxplots.png`         | Side-by-side box plot comparisons               |
| `categorical_distributions.png`  | Grouped bar charts for categorical features     |
| `psi_bar_chart.png`              | PSI severity chart with threshold lines         |

---

## 🚀 How to Run

```bash
# Navigate to Lab5
cd MLOPS-in-action/Lab5

# Install dependencies
pip install evidently ucimlrepo pandas numpy matplotlib seaborn scikit-learn scipy

# Run the script
python data_drift_analysis.py
```

---

## 🔮 Future Scope

- Integrate Evidently into a CI/CD or MLOps pipeline for continuous data quality checks.
- Automate report generation and email alerts for drift threshold breaches.
- Extend to **model performance drift** and **target drift** analysis.
- Add **embedding drift detection** for text-based features.
- Use PSI thresholds as automated retrain triggers in MLOps workflows.