# api.py
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

from feature_engineer import FeatureEngineer  # noqa: F401 – requerido por joblib

BASE_DIR = Path(__file__).resolve().parent
PREPOSTING_MODEL_PATH = BASE_DIR / "model_preposting_pipeline.joblib"


# ── Cargar modelo pre-publicacion ────────────────────────────────────────────
def _load_preposting():
    if not PREPOSTING_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontro model_preposting_pipeline.joblib en {BASE_DIR}. "
            "Ejecuta primero: python train_preposting_pipeline.py"
        )
    return joblib.load(PREPOSTING_MODEL_PATH)


_artifact = _load_preposting()
_pipeline = _artifact["full_pipeline"]
_feature_cols = _artifact["feature_cols_preposting"]
_metadata = _artifact["metadata"]


# ── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Viral Content Predictor",
    description="Predice si un post se hara viral usando solo features disponibles ANTES de publicar.",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    platform: str
    content_type: str
    topic: str
    language: str
    region: str
    post_datetime: str          # "YYYY-MM-DD HH:MM:SS"
    hashtags: str               # texto con los hashtags, ej: "#ai #ml #tech"
    sentiment_score: float      # -1.0 a 1.0


class PredictResponse(BaseModel):
    is_viral: bool
    label: str
    confidence: float           # probabilidad de ser viral (0-1)
    confidence_pct: str
    inputs_used: dict


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "online",
        "model": "pre-posting viral predictor",
        "docs": "/docs",
        "predict": "POST /predict-viral",
        "options": "/options",
    }


@app.get("/options")
def get_options():
    """Devuelve las opciones validas para cada dropdown del formulario."""
    return _metadata


@app.post("/predict-viral", response_model=PredictResponse)
def predict_viral(req: PredictRequest):
    """
    Predice si un post se hard viral usando solo features que se conocen
    ANTES de publicarlo (sin views, likes, comments, shares ni engagement_rate).
    """
    data = {
        "sentiment_score": [req.sentiment_score],
        "hashtags":        [req.hashtags],
        "post_datetime":   [req.post_datetime],
        "platform":        [req.platform],
        "content_type":    [req.content_type],
        "topic":           [req.topic],
        "language":        [req.language],
        "region":          [req.region],
    }

    df = pd.DataFrame(data)[_feature_cols]

    try:
        prediction = int(_pipeline.predict(df)[0])
        proba = float(_pipeline.predict_proba(df)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {exc}")

    return PredictResponse(
        is_viral=bool(prediction),
        label="Viral" if prediction == 1 else "No viral",
        confidence=round(proba, 4),
        confidence_pct=f"{proba * 100:.1f}%",
        inputs_used=req.model_dump(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
