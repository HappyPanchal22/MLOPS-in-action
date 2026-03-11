# Titanic Survival Predictor API 🚢

A FastAPI-based REST API that predicts whether a Titanic passenger would survive, powered by a **Random Forest Classifier**.

## Project Structure
```
fastapi_lab1/
├── model/
│   └── titanic_model.pkl
├── src/
│   ├── data.py       # Pydantic request/response models
│   ├── train.py      # Model training script
│   ├── predict.py    # Inference logic
│   └── main.py       # FastAPI app
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

## Train the Model

```bash
cd src
python train.py
```

## Run the API

```bash
uvicorn main:app --reload
```

API will be available at: `http://127.0.0.1:8000`  
Swagger docs at: `http://127.0.0.1:8000/docs`

## Sample Request

**POST** `/predict`

```json
{
  "pclass": 1,
  "sex": "female",
  "age": 29.0,
  "sibsp": 0,
  "parch": 0,
  "fare": 211.33,
  "embarked": "S"
}
```

**Response**

```json
{
  "survived": 1,
  "survival_label": "Survived",
  "survival_probability": 0.97
}
```

## Tech Stack
- **FastAPI** — API framework
- **scikit-learn** — Random Forest model
- **pandas** — Data preprocessing
- **uvicorn** — ASGI server
