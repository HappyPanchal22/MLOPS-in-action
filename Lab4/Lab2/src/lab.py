import pandas as pd
import numpy as np
import pickle
import base64
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# --- Task 1: Load Data ---
def load_data(ti, file_path: str = "/opt/airflow/dags/data/WineQT.csv"):
    """
    Loads the Wine Quality dataset, drops the Id column,
    and pushes raw data to XCom.
    """
    df = pd.read_csv(file_path)

    # Drop the Id column — not a feature
    df.drop(columns=['Id'], inplace=True)

    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nQuality score distribution:\n{df['quality'].value_counts().sort_index()}")

    serialized_data = base64.b64encode(pickle.dumps(df)).decode("ascii")
    ti.xcom_push(key='raw_data', value=serialized_data)


# --- Task 2: Preprocess Data ---
def preprocess_data(ti):
    """
    Pulls raw data, converts quality scores into binary labels
    (Good = 1 if quality >= 7, Bad = 0 otherwise), scales features,
    and splits into train/test sets.
    """
    serialized_data = ti.xcom_pull(key='raw_data', task_ids='load_data_task')
    df = pickle.loads(base64.b64decode(serialized_data))

    # Binary classification: Good (>=7) vs Bad (<7)
    df['quality_label'] = (df['quality'] >= 7).astype(int)

    good = df['quality_label'].sum()
    bad  = len(df) - good
    print(f"✅ Label distribution — Good (>=7): {good} | Bad (<7): {bad}")

    # Features and target
    X = df.drop(columns=['quality', 'quality_label'])
    y = df['quality_label']

    # Train/test split — stratified to preserve class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"✅ Train size: {X_train_scaled.shape} | Test size: {X_test_scaled.shape}")

    data_tuple = (X_train_scaled, X_test_scaled, y_train, y_test,
                  scaler, list(X.columns))
    serialized_output = base64.b64encode(pickle.dumps(data_tuple)).decode("ascii")
    ti.xcom_push(key='processed_data', value=serialized_output)


# --- Task 3: Train and Save Model ---
def train_save_model(ti):
    """
    Pulls processed data, uses GridSearchCV to find the best
    Decision Tree depth, trains the final model, and saves it to disk.
    """
    serialized_data = ti.xcom_pull(key='processed_data', task_ids='preprocess_data_task')
    (X_train_scaled, _, y_train, _, scaler, feature_names) = \
        pickle.loads(base64.b64decode(serialized_data))

    # Tune max_depth using cross-validation
    param_grid = {'max_depth': [3, 5, 7, 10, None]}
    dt = DecisionTreeClassifier(
        criterion='gini',
        class_weight='balanced',   # handles class imbalance (Good wines are fewer)
        random_state=42
    )
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_depth = grid_search.best_params_['max_depth']
    best_score = grid_search.best_score_
    print(f"✅ Best max_depth: {best_depth} | CV Accuracy: {best_score:.2%}")

    # Train final model with best params
    best_model = grid_search.best_estimator_
    print(f"✅ Decision Tree trained. Tree depth: {best_model.get_depth()}")

    # Save model, scaler, and feature names together
    model_dir = "/opt/airflow/dags/model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "wine_dt_model.sav")

    with open(model_path, "wb") as f:
        pickle.dump((best_model, scaler, feature_names), f)

    print(f"✅ Model saved to {model_path}")
    ti.xcom_push(key='model_path', value=model_path)


# --- Task 4: Evaluate Model ---
def evaluate_model(ti):
    """
    Loads the saved model, runs predictions on the test set,
    and prints accuracy, ROC-AUC, confusion matrix,
    classification report, and feature importances.
    """
    serialized_data = ti.xcom_pull(key='processed_data', task_ids='preprocess_data_task')
    (_, X_test_scaled, _, y_test, _, feature_names) = \
        pickle.loads(base64.b64decode(serialized_data))

    model_path = ti.xcom_pull(key='model_path', task_ids='train_save_model_task')
    best_model, _, feature_names = pickle.load(open(model_path, "rb"))

    # Predictions
    y_pred      = best_model.predict(X_test_scaled)
    y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_pred_prob)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred, target_names=['Bad (<7)', 'Good (>=7)']
    )

    print("\n========== 🍷 Wine Quality - Model Evaluation ==========")
    print(f"✅ Accuracy  : {accuracy:.2%}")
    print(f"✅ ROC-AUC   : {roc_auc:.4f}")
    print(f"✅ Tree Depth: {best_model.get_depth()}")
    print(f"\nConfusion Matrix (Rows=Actual, Cols=Predicted):")
    print(f"                Predicted Bad   Predicted Good")
    print(f"Actual Bad   :      {cm[0][0]:<10}  {cm[0][1]}")
    print(f"Actual Good  :      {cm[1][0]:<10}  {cm[1][1]}")
    print(f"\nClassification Report:\n{report}")

    # Feature importances
    importances = best_model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]

    print("\n📊 Feature Importances (most to least influential):")
    for i in sorted_idx:
        bar = "█" * int(importances[i] * 50)
        print(f"  {feature_names[i]:<25}: {importances[i]:.4f}  {bar}")
    print("=========================================================")
