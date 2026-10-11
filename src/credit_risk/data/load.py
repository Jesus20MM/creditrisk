"""Funciones de carga de datos."""

from pathlib import Path
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Carga un archivo CSV y devuelve un DataFrame."""
    return pd.read_csv(path)
