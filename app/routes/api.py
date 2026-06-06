from fastapi import APIRouter
from pydantic import BaseModel, Field
from helpers import response
from services.rst.predictor import predict_rst
from services.cbr.retrieve import retrieve
from services.cbr.retain import retain_new_case

router = APIRouter()

class PredictInput(BaseModel):
    studytime: int = Field(..., ge=1, le=4, description="Waktu belajar (1-4)")
    absences: int = Field(..., ge=0, description="Jumlah absen")
    failures: int = Field(..., ge=0, le=4, description="Jumlah kegagalan kelas (0-4)")
    famrel: int = Field(..., ge=1, le=5, description="Hubungan keluarga (1-5)")
    freetime: int = Field(..., ge=1, le=5, description="Waktu luang (1-5)")
    goout: int = Field(..., ge=1, le=5, description="Intensitas keluar rumah (1-5)")

class RetainInput(PredictInput):
    final_decision: str = Field(..., description="Keputusan final (Normal / Waspada / Bahaya)")


@router.get("/")
async def root():
    return response.success("Welcome to Alcohol Addiction Prediction API System")


@router.get("/health")
async def health():
    return response.success("Healthy")


@router.post("/predict")
async def predict_hybrid(request: PredictInput):
    try:
        raw_input = request.model_dump()

        rst_results = predict_rst(raw_input)
        
        cbr_results = retrieve(raw_input)

        target_classes = ["Normal", "Waspada", "Bahaya"]
        final_scores = {}
        
        for label in target_classes:
            sr = rst_results.get(label, 0.0)
            sc = cbr_results.get(label, 0.0)
            
            # Final Score = (SR * 0.6) + (SC * 0.4)
            score_kombinasi = (sr * 0.6) + (sc * 0.4)
            final_scores[label] = round(score_kombinasi, 4)

        # Pemilihan Keputusan dengan Skor Tertinggi
        final_decision = max(final_scores, key=lambda k: final_scores[k])
        highest_score = final_scores[final_decision]

        formatted_data = {
            "prediction": final_decision,
            "confidence_percentage": round(highest_score * 100, 2),
            "detail_scores": {
                "final_scores": final_scores,
                "rst_confidence": rst_results,
                "cbr_similarity": cbr_results
            }
        }

        return response.success("Proses prediksi hybrid berhasil diselesaikan", data=formatted_data)

    except Exception as e:
        return response.error(f"Gagal memproses prediksi: {str(e)}")


@router.post("/cbr/retain")
async def cbr_retain(input: RetainInput):
    try:
        # Mengambil input fitur saja dengan membuang kolom final_decision
        raw_features = input.model_dump(exclude={"final_decision"})
        
        # Panggil service retain milik CBR
        success = retain_new_case(raw_features, input.final_decision)
        if success:
            return response.success("Kasus baru berhasil di-retain ke dalam basis data CBR")
        
        return response.error("Proses penyimpanan kasus baru ke database gagal")
        
    except Exception as e:
        return response.error(f"Error pada proses retain data: {str(e)}")