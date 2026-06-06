from fastapi import APIRouter
from pydantic import BaseModel
from helpers import response
from services.cbr.retrieve import retrieve
from services.cbr.retain import retain_new_case

router = APIRouter()

class PredictInput(BaseModel):
    studytime: int
    absences: int
    failures: int
    famrel: int
    freetime: int
    goout: int

class RetainInput(BaseModel):
    studytime: int
    absences: int
    failures: int
    famrel: int
    freetime: int
    goout: int
    final_decision: str

@router.get("/")
async def root():
    return response.success("Welcome to FastAPI")

@router.get("/health")
async def health():
    return response.success("Healthy")

@router.post("/cbr/predict")
async def cbr_predict(input: PredictInput):
    try:
        raw_input = input.model_dump()
        cbr_scores = retrieve(raw_input)
        return response.success("CBR prediction successful", data={
            "cbr_scores": cbr_scores
        })
    except Exception as e:
        return response.error(str(e))

@router.post("/cbr/retain")
async def cbr_retain(input: RetainInput):
    try:
        raw_input = input.model_dump(exclude={"final_decision"})
        success = retain_new_case(raw_input, input.final_decision)
        if success:
            return response.success("Kasus baru berhasil disimpan")
        return response.error("Gagal menyimpan kasus")
    except Exception as e:
        return response.error(str(e))