from pydantic import BaseModel, Field
from enum import Enum

class SexEnum(str, Enum):
    male = "male"
    female = "female"

class EmbarkedEnum(str, Enum):
    C = "C"   # Cherbourg
    Q = "Q"   # Queenstown
    S = "S"   # Southampton

class TitanicInput(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)")
    sex: SexEnum = Field(..., description="Passenger sex: 'male' or 'female'")
    age: float = Field(..., ge=0.0, le=120.0, description="Passenger age in years")
    sibsp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    fare: float = Field(..., ge=0.0, description="Ticket fare in USD")
    embarked: EmbarkedEnum = Field(..., description="Port of embarkation: C, Q, or S")

    class Config:
        json_schema_extra = {
            "example": {
                "pclass": 3,
                "sex": "male",
                "age": 22.0,
                "sibsp": 1,
                "parch": 0,
                "fare": 7.25,
                "embarked": "S"
            }
        }

class TitanicResponse(BaseModel):
    survived: int = Field(..., description="0 = Did not survive, 1 = Survived")
    survival_label: str = Field(..., description="Human-readable survival result")
    survival_probability: float = Field(..., description="Model confidence for survival")