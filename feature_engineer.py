import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformador que extrae características temporales y de hashtags.

    Columnas de entrada requeridas:
        post_datetime (str o datetime), hashtags (str)

    Columnas nuevas que agrega:
        hour        - hora del post (0-23)
        dayofweek   - día de la semana (0 lunes … 6 domingo)
        n_hashtags  - cantidad de '#' en el campo hashtags
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["post_datetime"] = pd.to_datetime(X["post_datetime"])
        X["hour"] = X["post_datetime"].dt.hour
        X["dayofweek"] = X["post_datetime"].dt.dayofweek
        X["n_hashtags"] = X["hashtags"].fillna("").str.count("#")
        return X
