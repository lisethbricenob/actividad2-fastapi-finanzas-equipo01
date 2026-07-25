"""Carga del modelo y generación de predicciones financieras."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from financial_api import data as data_module
from financial_api import features as features_module


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "artifacts" / "model.joblib"
METADATA_PATH = ROOT_DIR / "artifacts" / "model_metadata.json"


class PredictionServiceError(RuntimeError):
    """Error controlado durante la carga o inferencia del modelo."""


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Carga y conserva en memoria el modelo serializado."""

    if not MODEL_PATH.exists():
        raise PredictionServiceError(
            f"No se encontró el modelo en: {MODEL_PATH}"
        )

    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        raise PredictionServiceError(
            f"No fue posible cargar el modelo: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def load_metadata() -> dict[str, Any]:
    """Carga los metadatos asociados al modelo."""

    if not METADATA_PATH.exists():
        raise PredictionServiceError(
            f"No se encontraron los metadatos en: {METADATA_PATH}"
        )

    try:
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionServiceError(
            f"No fue posible cargar los metadatos: {exc}"
        ) from exc


def get_latest_features(
    symbol: str,
    use_cached_data: bool = True,
) -> tuple[pd.Series, str]:
    """
    Obtiene la última fila completa de variables predictoras para un símbolo.

    Retorna:
        - La última fila de características.
        - La fuente utilizada: cache o local.
    """

    normalized_symbol = symbol.strip().upper()

    cache_path = data_module.RAW_DIR / f"{normalized_symbol}.csv"
    cache_existed = cache_path.exists() and use_cached_data

    symbol_data = data_module.get_symbol_data(
        symbol=normalized_symbol,
        use_cache=use_cached_data,
        refresh=not use_cached_data,
    )

    if symbol_data.empty:
        raise PredictionServiceError(
            f"No se encontraron datos para {normalized_symbol}."
        )

    dataset = symbol_data.copy()
    dataset.insert(0, "symbol", normalized_symbol)

    processed = features_module.build_features(dataset)

    if processed.empty:
        raise PredictionServiceError(
            f"No existen suficientes datos para calcular las variables de "
            f"{normalized_symbol}."
        )

    latest_row = processed.sort_values("date").iloc[-1]
    data_source = "cache" if cache_existed else "local"

    return latest_row, data_source


def predict_symbol(
    symbol: str,
    use_cached_data: bool = True,
) -> dict[str, Any]:
    """Genera la predicción de tendencia para un símbolo."""

    normalized_symbol = symbol.strip().upper()

    metadata = load_metadata()
    model = load_model()

    allowed_symbols = metadata.get("symbols", [])

    if normalized_symbol not in allowed_symbols:
        allowed_text = ", ".join(allowed_symbols)
        raise PredictionServiceError(
            f"Símbolo no soportado. Valores permitidos: {allowed_text}."
        )

    feature_columns = metadata.get("features", [])

    if not feature_columns:
        raise PredictionServiceError(
            "Los metadatos no contienen la lista de variables del modelo."
        )

    latest_row, data_source = get_latest_features(
        symbol=normalized_symbol,
        use_cached_data=use_cached_data,
    )

    missing_features = [
        column
        for column in feature_columns
        if column not in latest_row.index
    ]

    if missing_features:
        raise PredictionServiceError(
            "Faltan variables requeridas por el modelo: "
            + ", ".join(missing_features)
        )

    model_input = pd.DataFrame(
        [latest_row[feature_columns].to_dict()],
        columns=feature_columns,
    )

    try:
        probability_up = float(model.predict_proba(model_input)[0, 1])
    except Exception as exc:
        raise PredictionServiceError(
            f"No fue posible generar la predicción: {exc}"
        ) from exc

    prediction = "up" if probability_up >= 0.5 else "down"

    return {
        "symbol": normalized_symbol,
        "prediction": prediction,
        "probability_up": round(probability_up, 4),
        "model_version": metadata["model_version"],
        "prediction_horizon": metadata["prediction_horizon"],
        "data_source": data_source,
    }