import pandas as pd
import pytest
from credit_risk.data.validate import validate_columns


def test_validate_columns_accepts_required_columns():
    df = pd.DataFrame({"customer_id": [1], "default": [0]})
    validate_columns(df, ["customer_id", "default"])


def test_validate_columns_rejects_missing_columns():
    df = pd.DataFrame({"customer_id": [1]})
    with pytest.raises(ValueError):
        validate_columns(df, ["default"])
