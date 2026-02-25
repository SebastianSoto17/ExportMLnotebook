"""
verify_pipeline.py
------------------
Verifica que:
  1. El archivo model_realistic_pipeline.joblib existe y se carga sin errores.
  2. El modelo tiene los componentes esperados.
  3. Se pueden realizar predicciones con datos de muestra.

Ejecución:
    python verify_pipeline.py
"""

import sys
from pathlib import Path

import pandas as pd

from load_realistic_pipeline import get_model, get_feature_cols, load_pipeline

MODEL_PATH = Path(__file__).resolve().parent / "model_realistic_pipeline.joblib"

SAMPLE_DATA = {
    "views":            [1_000,   500_000],
    "likes":            [50,      45_000],
    "comments":         [5,       3_200],
    "shares":           [3,       8_100],
    "engagement_rate":  [0.05,    0.09],
    "sentiment_score":  [-0.2,    0.75],
    "hashtags":         ["#test", "#viral #trending #fyp"],
    "post_datetime":    ["2025-03-15 08:00:00", "2025-06-21 18:30:00"],
    "platform":         ["Instagram",           "TikTok"],
    "content_type":     ["image",               "video"],
    "topic":            ["lifestyle",           "entertainment"],
    "language":         ["es",                  "en"],
    "region":           ["latam",               "north_america"],
}

LABEL = {0: "No viral [X]", 1: "Viral [OK]"}


def check(condition: bool, msg_ok: str, msg_fail: str):
    if condition:
        print(f"  [OK]  {msg_ok}")
    else:
        print(f"  [FAIL] {msg_fail}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  Verificación de model_realistic_pipeline.joblib")
    print("=" * 60)

    # ── 1. Archivo existe ─────────────────────────────────────────────────────
    print("\n[1] Comprobando existencia del archivo …")
    check(
        MODEL_PATH.is_file(),
        f"Archivo encontrado ({MODEL_PATH.stat().st_size / 1024:.1f} KB)",
        f"Archivo NO encontrado en {MODEL_PATH}",
    )

    # ── 2. Se carga correctamente ─────────────────────────────────────────────
    print("\n[2] Cargando artefacto …")
    try:
        artifact = load_pipeline(MODEL_PATH)
        check(isinstance(artifact, dict), "Artefacto cargado como dict", "")
    except Exception as exc:
        print(f"  [FAIL] Error al cargar: {exc}")
        sys.exit(1)

    expected_keys = {"model_realistic", "preprocessor_realistic", "feature_cols_realistic"}
    present = expected_keys.issubset(artifact.keys())
    check(present, f"Claves encontradas: {list(artifact.keys())}", f"Faltan claves: {expected_keys - artifact.keys()}")

    # ── 3. Columnas de entrada ────────────────────────────────────────────────
    print("\n[3] Columnas de entrada …")
    feature_cols = get_feature_cols(MODEL_PATH)
    check(
        bool(feature_cols),
        f"{len(feature_cols)} columnas: {feature_cols}",
        "No se encontraron columnas de entrada",
    )

    # ── 4. Predicciones ───────────────────────────────────────────────────────
    print("\n[4] Haciendo predicciones con datos de muestra …")
    try:
        pipeline = get_model(MODEL_PATH)
        df = pd.DataFrame(SAMPLE_DATA)[feature_cols]
        predictions = pipeline.predict(df)
        check(len(predictions) == len(df), f"Predicciones obtenidas: {predictions.tolist()}", "")

        print("\n  Resultados detallados:")
        print(f"  {'#':<4} {'Platform':<12} {'Content':<8} {'Views':>8}  {'Prediction'}")
        print("  " + "-" * 52)
        for i, pred in enumerate(predictions):
            row = df.iloc[i]
            print(
                f"  {i+1:<4} {SAMPLE_DATA['platform'][i]:<12} "
                f"{SAMPLE_DATA['content_type'][i]:<8} "
                f"{SAMPLE_DATA['views'][i]:>8,}  {LABEL.get(pred, str(pred))}"
            )
    except Exception as exc:
        print(f"  [FAIL] Error al predecir: {exc}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Todo correcto. El pipeline esta listo para produccion.")
    print("=" * 60)


if __name__ == "__main__":
    main()
