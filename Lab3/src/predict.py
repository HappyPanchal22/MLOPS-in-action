import pickle
import pandas as pd
from pathlib import Path

# Resolve model path relative to this file
MODEL_PATH = Path(__file__).parent.parent / "model" / "titanic_model.pkl"

# Encoding maps must match what train.py used (LabelEncoder ordering)
SEX_MAP = {"female": 0, "male": 1}
EMBARKED_MAP = {"C": 0, "Q": 1, "S": 2}

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please run train.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# Load once at import time (reused across requests)
model = load_model()

def predict(pclass: int, sex: str, age: float,
            sibsp: int, parch: int, fare: float, embarked: str) -> dict:
    """
    Encode inputs, run inference, and return prediction + probability.
    """
    sex_encoded = SEX_MAP[sex]
    embarked_encoded = EMBARKED_MAP[embarked]

    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex_encoded,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked_encoded
    }])

    prediction = int(model.predict(input_df)[0])
    proba = float(model.predict_proba(input_df)[0][1])  # P(survived=1)

    return {
        "survived": prediction,
        "survival_label": "Survived" if prediction == 1 else "Did Not Survive",
        "survival_probability": round(proba, 4)
    }