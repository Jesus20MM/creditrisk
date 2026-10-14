"""Funciones sencillas para separar variables y construir preprocesadores."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construye un preprocesador según los tipos de columnas de X.

    Las columnas numéricas se imputan con la mediana y se escalan.
    Las categóricas se imputan con la categoría más frecuente y se codifican.
    El preprocesador debe ajustarse únicamente con los datos de entrenamiento.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                make_column_selector(dtype_include="number"),
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_exclude="number"),
            ),
        ],
        remainder="drop",
    )
