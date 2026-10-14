import pandas as pd

from proyecto_ml.preprocessing import build_preprocessor


def test_preprocessor_handles_numeric_and_categorical_columns():
    X = pd.DataFrame(
        {
            "age": [20, None, 35],
            "city": ["A", "B", None],
        }
    )
    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] >= 2
