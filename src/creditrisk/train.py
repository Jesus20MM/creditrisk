"""Ejemplo de entrenamiento supervisado para datos tabulares.

Antes de ejecutar:
- Coloca un archivo en data/raw/dataset.csv.
- Cambia TARGET_COLUMN por el nombre real de tu variable objetivo.
- Elige clasificación o regresión según la tarea.
"""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from proyecto_ml.preprocessing import build_preprocessor

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "dataset.csv"
TARGET_COLUMN = "target"
TASK = "classification"  # Cambia a "regression" si corresponde.
RANDOM_STATE = 42


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. Coloca allí tu CSV y actualiza la configuración."
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"No existe la columna objetivo {TARGET_COLUMN!r}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    df = df.dropna(subset=[TARGET_COLUMN])
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    stratify = y if TASK == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    preprocessor = build_preprocessor(X_train)

    if TASK == "classification":
        baseline_model = DummyClassifier(strategy="most_frequent")
        model = LogisticRegression(max_iter=1000)
    elif TASK == "regression":
        baseline_model = DummyRegressor(strategy="mean")
        model = Ridge()
    else:
        raise ValueError("TASK debe ser 'classification' o 'regression'.")

    baseline = Pipeline(
        steps=[("preprocessor", build_preprocessor(X_train)), ("model", baseline_model)]
    )
    candidate = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    baseline.fit(X_train, y_train)
    candidate.fit(X_train, y_train)

    baseline_pred = baseline.predict(X_test)
    candidate_pred = candidate.predict(X_test)

    if TASK == "classification":
        print(f"Baseline accuracy: {accuracy_score(y_test, baseline_pred):.4f}")
        print(f"Model accuracy:    {accuracy_score(y_test, candidate_pred):.4f}")
    else:
        print(f"Baseline MAE: {mean_absolute_error(y_test, baseline_pred):.4f}")
        print(f"Model MAE:    {mean_absolute_error(y_test, candidate_pred):.4f}")
        print(f"Model R²:     {r2_score(y_test, candidate_pred):.4f}")

    print("\nImportante: selecciona métricas adecuadas a tu problema y analiza los errores.")


if __name__ == "__main__":
    main()
