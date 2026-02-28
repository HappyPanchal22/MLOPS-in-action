from fastapi import FastAPI, HTTPException
from data import TitanicInput, TitanicResponse
from predict import predict

app = FastAPI(
    title="Titanic Survival Predictor",
    description="Predicts whether a Titanic passenger would survive using a Random Forest model.",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Titanic Survival Prediction API is running 🚢"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=TitanicResponse)
async def predict_survival(passenger: TitanicInput):
    """
    Predict survival for a given passenger.
    - **pclass**: Ticket class (1, 2, or 3)
    - **sex**: 'male' or 'female'
    - **age**: Age in years
    - **sibsp**: # of siblings/spouses aboard
    - **parch**: # of parents/children aboard
    - **fare**: Passenger fare
    - **embarked**: Port of embarkation — C (Cherbourg), Q (Queenstown), S (Southampton)
    """
    try:
        result = predict(
            pclass=passenger.pclass,
            sex=passenger.sex.value,
            age=passenger.age,
            sibsp=passenger.sibsp,
            parch=passenger.parch,
            fare=passenger.fare,
            embarked=passenger.embarked.value
        )
        return TitanicResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")