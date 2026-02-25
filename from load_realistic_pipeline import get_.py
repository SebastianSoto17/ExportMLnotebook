import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformador para crear nuevas características a partir de las columnas originales."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Aseguramos formato datetime
        X["post_datetime"] = pd.to_datetime(X["post_datetime"])

        # Variables temporales
        X["hour"] = X["post_datetime"].dt.hour
        X["dayofweek"] = X["post_datetime"].dt.dayofweek  # 0 = lunes, 6 = domingo

        # Contador simple de hashtags
        X["n_hashtags"] = X["hashtags"].fillna("").str.count("#")

        return X


from load_realistic_pipeline import get_model, predict

# Ejemplo de uso (requiere definir X_nuevos_datos como DataFrame con las columnas esperadas)
pipeline = get_model()

# Hacer predicciones
# y_pred = pipeline.predict(X_nuevos_datos)
# o
# y_pred = predict(X_nuevos_datos)
