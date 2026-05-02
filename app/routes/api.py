from fastapi import APIRouter
from helpers import response

router = APIRouter()

@router.get("/")
async def root():
	return response.success("Welcome to FastAPI")

@router.get("/health")
async def health():
	return response.success("Healthy")
