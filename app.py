"""FastAPI endpoint for the saved student score regression model."""
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "mental_health_model.joblib"
app = FastAPI(title="Student Mental Health Score Predictor", version="1.0.0",
              description="Predicts the dataset's mental-health score. This is not a clinical assessment.")

class StudentInput(BaseModel):
    Age: int = Field(ge=10, le=120)
    Gender: str
    Country: str
    Academic_Level: str
    Most_Used_Platform: str
    Purpose_Of_Use: str
    Avg_Daily_Usage_Hours: float = Field(ge=0, le=24)
    Daily_Unlocks: int = Field(ge=0)
    Study_Hours: float = Field(ge=0, le=24)
    Physical_Activity_Hours: float = Field(ge=0, le=24)
    Sleep_Hours_Per_Night: float = Field(ge=0, le=24)
    Stress_Level: Literal["Low", "Medium", "High", "Very High"]

class Prediction(BaseModel):
    predicted_mental_health_score: float
    note: str


def load_artifact():
    if not ARTIFACT_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact not found. Run `python train_model.py` first.")
    return joblib.load(ARTIFACT_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": ARTIFACT_PATH.exists()}

@app.post("/predict", response_model=Prediction)
def predict(student: StudentInput):
    artifact = load_artifact()
    row = pd.DataFrame([student.model_dump() if hasattr(student, "model_dump") else student.dict()])
    prediction = float(artifact["model"].predict(row)[0])
    return Prediction(predicted_mental_health_score=round(prediction, 2),
                      note="A model estimate from this dataset; not a diagnosis or clinical advice.")
