"""
load_realistic_pipeline.py
--------------------------
Carga el artefacto model_realistic_pipeline.joblib y expone helpers para
obtener el pipeline completo, sus componentes y hacer predicciones.
"""

import joblib
from pathlib import Path
from typing import Any, Dict, Tuple

# Importar FeatureEngineer desde su propio módulo garantiza que joblib/pickle
# siempre pueda resolver la clase al deserializar, sin importar desde dónde
# se llame este archivo.
from feature_engineer import FeatureEngineer  # noqa: F401  (re-export implícito)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_realistic_pipeline.joblib"


def load_pipeline(model_path: Path | str = MODEL_PATH) -> Dict[str, Any]:
    """
    Carga el artefacto completo (.joblib) y devuelve el dict con claves:
        model_realistic, preprocessor_realistic, feature_cols_realistic,
        full_pipeline  (pipeline sklearn completo, si existe)
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError(
            "El .joblib debe contener un dict con las claves "
            "'model_realistic', 'preprocessor_realistic' y 'feature_cols_realistic'."
        )
    return artifact


def get_components(model_path: Path | str = MODEL_PATH) -> Tuple[Any, Any, list]:
    """Devuelve (model_realistic, preprocessor_realistic, feature_cols_realistic)."""
    artifact = load_pipeline(model_path)
    return (
        artifact["model_realistic"],
        artifact["preprocessor_realistic"],
        artifact["feature_cols_realistic"],
    )


def get_model(model_path: Path | str = MODEL_PATH):
    """
    Devuelve el pipeline sklearn completo (feature_engineering → preprocessor → classifier).
    Si el artefacto tiene 'full_pipeline', lo usa directamente; si no, construye
    un wrapper equivalente.
    """
    artifact = load_pipeline(model_path)

    if "full_pipeline" in artifact:
        return artifact["full_pipeline"]

    # Compatibilidad con artefactos antiguos que no guardaban full_pipeline
    from sklearn.pipeline import Pipeline

    model = artifact["model_realistic"]
    pre = artifact["preprocessor_realistic"]

    class _WrappedPipeline:
        def predict(self, X):
            return model.predict(pre.transform(X))

    return _WrappedPipeline()


def get_feature_cols(model_path: Path | str = MODEL_PATH) -> list:
    """Devuelve la lista de columnas de entrada del modelo."""
    _, _, feature_cols = get_components(model_path)
    return feature_cols


def predict(inputs, model_path: Path | str = MODEL_PATH):
    """Hace predicciones directamente sobre un DataFrame con las columnas crudas."""
    pipeline = get_model(model_path)
    return pipeline.predict(inputs)
