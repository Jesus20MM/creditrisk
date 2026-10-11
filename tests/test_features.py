import pandas as pd
from credit_risk.features.build_features import build_features


def test_build_features_returns_copy_with_same_columns():
    df = pd.DataFrame({"income": [1000]})
    result = build_features(df)
    assert list(result.columns) == ["income"]
    assert result is not df
