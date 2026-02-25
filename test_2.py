import joblib
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_realistic_pipeline.joblib"


def load_pipeline(model_path: Path | str = MODEL_PATH) -> Any:
    """
    Carga el pipeline completo (preprocesamiento + modelo) desde un archivo .joblib.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo del modelo en: {model_path}")

    pipeline = joblib.load(model_path)
    return pipeline


def get_model() -> Any:
    """
    Carga el pipeline por defecto `model_realistic_pipeline.joblib`.
    """
    return load_pipeline()


def predict(inputs) -> Any:
    """
    Realiza predicciones usando el pipeline cargado.
    """
    pipeline = get_model()
    return pipeline.predict(inputs)