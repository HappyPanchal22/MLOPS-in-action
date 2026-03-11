import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def load_and_preprocess():
    # Load Titanic dataset from seaborn's hosted CSV (no extra download needed)
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    # --- Feature Engineering ---
    # Select relevant features
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    target = "Survived"

    df = df[features + [target]].copy()

    # Fill missing values
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])       # male=1, female=0
    df["Embarked"] = le.fit_transform(df["Embarked"])  # C=0, Q=1, S=2

    X = df[features]
    y = df[target]
    return X, y

def train():
    X, y = load_and_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model trained successfully!")
    print(f"📊 Test Accuracy: {acc:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"]))

    # Save model
    os.makedirs("../model", exist_ok=True)
    with open("../model/titanic_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("💾 Model saved to ../model/titanic_model.pkl")

if __name__ == "__main__":
    train()