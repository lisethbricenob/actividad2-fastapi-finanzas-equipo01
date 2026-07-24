"""
Entrenamiento y serialización del modelo (Rol 2: Datos, features y modelo).

Tarea: clasificación de tendencia (retorno del día siguiente positivo/negativo).
Modelo: RandomForestClassifier — sencillo, robusto y sin necesidad de escalado.

Salidas (requisitos 5 y 6 del enunciado):
    artifacts/model.joblib          -> modelo serializado con joblib
    artifacts/model_metadata.json   -> metadatos (fecha, símbolos, métrica, horizonte)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from financial_api import data as data_module
from financial_api import features as features_module

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

MODEL_VERSION = "random_forest_v1"
PREDICTION_HORIZON = "next_day"
TEST_FRACTION = 0.2
RANDOM_STATE = 42


def _time_aware_split(features, feature_cols, target_col, test_fraction):
    """
    Partición temporal por símbolo: el último tramo de cada activo se reserva
    para prueba. Evita fuga de información que produciría un split aleatorio en
    series de tiempo.
    """
    train_parts, test_parts = [], []
    for _, group in features.groupby("symbol", sort=False):
        group = group.sort_values("date")
        cut = int(len(group) * (1 - test_fraction))
        train_parts.append(group.iloc[:cut])
        test_parts.append(group.iloc[cut:])

    import pandas as pd
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return (
        train[feature_cols], train[target_col],
        test[feature_cols], test[target_col],
    )


def train_model(refresh_data: bool = False) -> dict:
    """Ejecuta el flujo completo: datos -> features -> entrenamiento -> artefactos."""
    dataset = data_module.build_raw_dataset(refresh=refresh_data)
    features = features_module.build_features(dataset)
    features_module.save_features(features)

    feature_cols = features_module.FEATURE_COLUMNS
    target_col = features_module.TARGET_COLUMN

    X_train, y_train, X_test, y_test = _time_aware_split(
        features, feature_cols, target_col, TEST_FRACTION
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Modelo serializado en %s", MODEL_PATH)

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "RandomForestClassifier",
        "task": "trend_classification",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "symbols": list(data_module.SYMBOLS),
        "features": feature_cols,
        "target": target_col,
        "prediction_horizon": PREDICTION_HORIZON,
        "primary_metric": "roc_auc",
        "metrics": metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    logger.info("Metadatos escritos en %s", METADATA_PATH)
    return metadata


def main() -> None:
    """Punto de entrada CLI: `python -m financial_api.train`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    metadata = train_model()
    print("Entrenamiento completado.")
    print(json.dumps(metadata["metrics"], indent=2))
    print(f"Modelo:    {MODEL_PATH}")
    print(f"Metadatos: {METADATA_PATH}")


if __name__ == "__main__":
    main()
