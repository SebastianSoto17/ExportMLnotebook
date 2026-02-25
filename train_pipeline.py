"""
train_pipeline.py
-----------------
Entrena un pipeline de clasificación (Random Forest) sobre el dataset de
contenido viral en redes sociales y serializa el artefacto completo en
model_realistic_pipeline.joblib.

Ejecución:
    python train_pipeline.py

El artefacto guardado es un dict con las claves:
    model_realistic        -> clasificador entrenado
    preprocessor_realistic -> ColumnTransformer ajustado
    feature_cols_realistic -> lista de columnas de entrada (X)
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineer import FeatureEngineer

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "social_media_viral_content_dataset.csv"
MODEL_PATH = BASE_DIR / "model_realistic_pipeline.joblib"

# ── Columnas ──────────────────────────────────────────────────────────────────
RAW_FEATURE_COLS = [
    "views", "likes", "comments", "shares",
    "engagement_rate", "sentiment_score",
    "hashtags", "post_datetime",
    "platform", "content_type", "topic", "language", "region",
]
TARGET_COL = "is_viral"

# Después de FeatureEngineer las columnas numéricas y categóricas cambian:
NUMERIC_COLS = [
    "views", "likes", "comments", "shares",
    "engagement_rate", "sentiment_score",
    "hour", "dayofweek", "n_hashtags",
]
CATEGORICAL_COLS = ["platform", "content_type", "topic", "language", "region"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def build_full_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )


def main():
    print(f"[1/4] Cargando datos desde {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"      Filas: {len(df):,}  |  Columnas: {df.columns.tolist()}")

    X = df[RAW_FEATURE_COLS]
    y = df[TARGET_COL]

    print("[2/4] Dividiendo en train/test (80/20) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[3/4] Entrenando pipeline ...")
    pipeline = build_full_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\n-- Metricas en test --------------------------------------------------")
    print(classification_report(y_test, y_pred, target_names=["No viral", "Viral"]))

    print(f"[4/4] Serializando artefacto en {MODEL_PATH} ...")
    artifact = {
        "model_realistic": pipeline.named_steps["classifier"],
        "preprocessor_realistic": Pipeline(
            steps=[
                ("feature_engineering", pipeline.named_steps["feature_engineering"]),
                ("preprocessor", pipeline.named_steps["preprocessor"]),
            ]
        ),
        "feature_cols_realistic": RAW_FEATURE_COLS,
        "full_pipeline": pipeline,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"      OK - Modelo guardado ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
