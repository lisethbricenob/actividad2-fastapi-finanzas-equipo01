"""
Construcción de variables predictivas (Rol 2: Datos, features y modelo).

Genera, a partir del OHLCV crudo, las variables que exige el enunciado:
retornos, medias móviles, volatilidad y rezagos. Define además la variable
objetivo para la tarea de CLASIFICACIÓN DE TENDENCIA: predecir si el retorno
del día siguiente será positivo (1) o negativo (0).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from financial_api import data as data_module

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "features.parquet"

# Ventanas parametrizadas para que features y train compartan la misma fuente.
MA_WINDOWS = (5, 10, 20)
VOL_WINDOWS = (5, 10)
LAGS = (1, 2, 3)

# Lista canónica de features que consumirá el modelo (usada también en predict).
FEATURE_COLUMNS: list[str] = (
    ["return_1d"]
    + [f"ma_{w}" for w in MA_WINDOWS]
    + [f"ma_ratio_{w}" for w in MA_WINDOWS]
    + [f"volatility_{w}" for w in VOL_WINDOWS]
    + [f"return_lag_{lag}" for lag in LAGS]
)

TARGET_COLUMN = "target_up"


def _features_for_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todas las features para un único símbolo ya ordenado por fecha."""
    out = df.sort_values("date").copy()
    close = out["close"]

    # Retorno diario simple.
    out["return_1d"] = close.pct_change()

    # Medias móviles y su cociente respecto al precio (señal de tendencia).
    for w in MA_WINDOWS:
        out[f"ma_{w}"] = close.rolling(w).mean()
        out[f"ma_ratio_{w}"] = close / out[f"ma_{w}"]

    # Volatilidad: desviación estándar móvil del retorno diario.
    for w in VOL_WINDOWS:
        out[f"volatility_{w}"] = out["return_1d"].rolling(w).std()

    # Rezagos del retorno (información autoregresiva).
    for lag in LAGS:
        out[f"return_lag_{lag}"] = out["return_1d"].shift(lag)

    # Objetivo: signo del retorno del DÍA SIGUIENTE (sin fuga temporal).
    out[TARGET_COLUMN] = (close.shift(-1) > close).astype("int")

    return out


def build_features(dataset: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Construye el dataset de features consolidado para todos los símbolos.
    Si no se pasa `dataset`, lo obtiene de data.build_raw_dataset().
    """
    if dataset is None:
        dataset = data_module.build_raw_dataset()

    frames = [
        _features_for_symbol(group)
        for _, group in dataset.groupby("symbol", sort=False)
    ]
    features = pd.concat(frames, ignore_index=True)

    # Se descartan filas con NaN generados por rolling/shift y la última fila
    # de cada símbolo (target futuro desconocido).
    required = ["symbol", "date", *FEATURE_COLUMNS, TARGET_COLUMN]
    features = features[required].dropna().reset_index(drop=True)
    return features


def save_features(features: pd.DataFrame, path: Path = PROCESSED_PATH) -> Path:
    """Persiste las features procesadas para reproducibilidad."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        features.to_parquet(path, index=False)
    except Exception:  # noqa: BLE001 - si falta pyarrow, se cae a CSV
        path = path.with_suffix(".csv")
        features.to_csv(path, index=False)
    logger.info("Features guardadas en %s (%d filas).", path, len(features))
    return path


def main() -> None:
    """Punto de entrada CLI: `python -m financial_api.features`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    features = build_features()
    out_path = save_features(features)
    print(f"Features: {len(features)} filas, {len(FEATURE_COLUMNS)} variables.")
    print(f"Balance de clases:\n{features[TARGET_COLUMN].value_counts(normalize=True)}")
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()
