"""Métricas para evaluar modelos."""

from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calcula métricas básicas de regresión."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def classification_auc(y_true, y_score) -> float:
    """Calcula ROC-AUC a partir de etiquetas reales y scores/probabilidades."""
    return float(roc_auc_score(y_true, y_score))
