"""Validaciones básicas de estructura y calidad de datos."""

import pandas as pd


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Lanza ValueError si faltan columnas requeridas."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
