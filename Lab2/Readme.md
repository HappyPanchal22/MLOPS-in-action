Here is your **properly formatted, clean, copy-paste ready README.md** file:

---

# 🍷 Vinous-ML: Automated Wine Quality Analytics

## End-to-End ML Orchestration with Apache Airflow & Docker

Vinous-ML is an automated Machine Learning pipeline that classifies wine quality based on chemical properties. Using the **WineQT dataset**, this project demonstrates full ML lifecycle orchestration using **Apache Airflow**, **Docker**, and **Scikit-learn** — from ingestion to evaluation.

---

## 🎯 Project Objective

The goal is to move beyond regression and implement a robust **Binary Classification** model.

### 🔄 Target Transformation

Original quality scores (1–10) are converted into binary labels:

* **Good (1)** → Quality ≥ 7
* **Bad (0)** → Quality < 7

### 🤖 Model Strategy

* **Algorithm:** Decision Tree Classifier
* **Optimization:** `GridSearchCV` for hyperparameter tuning
* **Handling Imbalance:** Stratified splitting + depth regularization

This ensures better generalization and avoids overfitting on imbalanced wine quality distributions.

---

## 📂 Project Structure

Ensure your project directory is structured exactly as shown below for proper Docker volume mapping:

```
Lab-2/
│   ├── winedag.py               # DAG Definition (Orchestrator)
│   ├── data/
│   │   └── WineQT.csv           # Raw Dataset
│   ├── model/
│   │   └── wine_dt_model.sav    # Optimized Pickle File
│   └── src/
│       └── lab.py               # ML Logic (Preprocessing & Training)
├── .env                         # User UID/GID Configuration
└── docker-compose.yaml          # Multi-container Setup
```

---

## 🛠 Setup & Installation

### 1️⃣ Environment Preparation

Initialize required folders and configure Airflow UID to prevent permission conflicts:

```bash
mkdir -p ./dags/data ./dags/model ./dags/src ./logs ./plugins
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

---

### 2️⃣ Dependency Configuration

Update your `docker-compose.yaml` file to include required scientific libraries:

```yaml
_PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- pandas scikit-learn numpy}
```

---

### 3️⃣ Launching the Airflow Cluster

Initialize Airflow metadata:

docker compose up airflow-init


Start all services:

docker compose up -d

Access the Airflow UI:

http://localhost:8080

**Default credentials:**

* Username: `airflow2`
* Password: `airflow2`

---

## 🔄 Pipeline Workflow

The DAG defined in `winedag.py` executes the following tasks in strict sequence:

### 1️⃣ Setup — `setup_directories`

Ensures the model export directory exists on the host machine.

### 2️⃣ Ingestion — `load_data_task`

* Loads the WineQT dataset
* Performs initial cleaning

### 3️⃣ ETL — `preprocess_data_task`

* Applies binary labeling
* Performs stratified train/test split
* Applies `StandardScaler` feature normalization

### 4️⃣ Optimization — `train_save_model_task`

* Executes `GridSearchCV`
* Tunes `max_depth` to prevent overfitting
* Saves the best model as a `.sav` file

### 5️⃣ Evaluation — `evaluate_model_task`

* Reloads optimized model
* Generates:

  * Confusion Matrix
  * ROC-AUC Score
  * ASCII-rendered Feature Importance chart

---

## 📊 How to Analyze Results

After DAG execution (all tasks turn **dark green**):

1. Click on **`evaluate_model_task`**
2. Open the **Logs** tab
3. Review:

   * Model accuracy metrics
   * ROC-AUC score
   * Feature Importance ranking (Alcohol, Volatile Acidity, etc.)

This allows performance auditing directly inside Airflow logs.

---

## 🧹 Cleanup

To stop and remove all containers:

docker compose down

---

## 🚀 Tech Stack

* Apache Airflow
* Docker & Docker Compose
* Python 3
* Scikit-learn
* Pandas
* NumPy


