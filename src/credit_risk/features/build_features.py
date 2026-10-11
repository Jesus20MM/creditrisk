"""Funciones de ingeniería de variables."""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Punto de entrada para construir variables; personalizar según los datos."""
    return df.copy()
