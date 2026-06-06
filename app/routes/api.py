from fastapi import APIRouter
from pydantic import BaseModel
from helpers import response
from services.rst.predictor import predict_rst

router = APIRouter()

class PredictRSTRequest(BaseModel):
    studytime: int
    absences: int
    failures: int
    famrel: int
    freetime: int
    goout: int

@router.get("/")
async def root():
    return response.success("Welcome to FastAPI")

@router.get("/health")
async def health():
    return response.success("Healthy")

@router.post("/predict-rst")
async def predict_rst_route(request: PredictRSTRequest):
    result = predict_rst(request.model_dump())
    return response.success("Berhasil memprediksi RST", data=result)