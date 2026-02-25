"""
train_preposting_pipeline.py
----------------------------
Entrena un modelo de clasificacion usando SOLO features conocibles
ANTES de publicar el contenido (sin views, likes, comments, shares ni
engagement_rate, que solo existen despues de publicar).

Features usadas:
    platform, content_type, topic, language, region  -> categoricas
    post_datetime  -> hora y dia de semana (feature engineering)
    hashtags       -> numero de hashtags (#)
    sentiment_score -> sentimiento del texto del post (-1 a 1)

Target: is_viral (0 = No viral, 1 = Viral)

Ejecucion:
    python train_preposting_pipeline.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineer import FeatureEngineer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "social_media_viral_content_dataset.csv"
MODEL_PATH = BASE_DIR / "model_preposting_pipeline.joblib"

# Features validas ANTES de publicar
RAW_FEATURE_COLS = [
    "sentiment_score",
    "hashtags",
    "post_datetime",
    "platform",
    "content_type",
    "topic",
    "language",
    "region",
]
TARGET_COL = "is_viral"

# Despues de FeatureEngineer se agregan: hour, dayofweek, n_hashtags
NUMERIC_COLS = ["sentiment_score", "hour", "dayofweek", "n_hashtags"]
CATEGORICAL_COLS = ["platform", "content_type", "topic", "language", "region"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42,
            )),
        ]
    )


def main():
    print("[1/4] Cargando dataset ...")
    df = pd.read_csv(DATA_PATH)
    print(f"      {len(df):,} filas | target: is_viral {df['is_viral'].value_counts().to_dict()}")

    X = df[RAW_FEATURE_COLS]
    y = df[TARGET_COL]

    print("[2/4] Dividiendo train/test (80/20) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[3/4] Entrenando Gradient Boosting con features pre-publicacion ...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print("\n-- Metricas en test --------------------------------------------------")
    print(classification_report(y_test, y_pred, target_names=["No viral", "Viral"]))

    print(f"[4/4] Guardando artefacto en {MODEL_PATH} ...")
    artifact = {
        "model_preposting": pipeline.named_steps["classifier"],
        "preprocessor_preposting": Pipeline(
            steps=[
                ("feature_engineering", pipeline.named_steps["feature_engineering"]),
                ("preprocessor", pipeline.named_steps["preprocessor"]),
            ]
        ),
        "feature_cols_preposting": RAW_FEATURE_COLS,
        "full_pipeline": pipeline,
        "metadata": {
            "platforms": ["Instagram", "TikTok", "X", "YouTube Shorts"],
            "content_types": ["carousel", "image", "text", "video"],
            "topics": ["Education", "Entertainment", "Lifestyle", "Politics", "Sports", "Technology"],
            "languages": ["en", "es", "fr", "hi", "ur"],
            "regions": ["Brazil", "India", "Pakistan", "UK", "US"],
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"      OK - {MODEL_PATH.stat().st_size / 1024:.1f} KB guardados")
    print()
    print("NOTA: Este modelo usa SOLO features conocibles antes de publicar.")
    print("      El modelo anterior (model_realistic_pipeline.joblib) usaba")
    print("      views/likes/comments/shares que solo existen DESPUES de publicar,")
    print("      lo que lo hace inutilizable para prediccion real pre-publicacion.")


if __name__ == "__main__":
    main()
