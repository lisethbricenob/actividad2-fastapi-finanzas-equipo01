"""
Ingesta de datos históricos de mercado (Rol 2: Datos, features y modelo).

Estrategia de reproducibilidad (requisito 10 del enunciado):

    1. Si existe caché local en data/raw/, se usa esa copia.
    2. Si no hay caché, se intenta descargar con yfinance.
    3. Si yfinance no está disponible (sin internet durante la evaluación),
       se genera un dataset sintético determinista para que la API y las
       pruebas puedan ejecutarse igualmente.

Así el proyecto NUNCA depende exclusivamente de internet para funcionar.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Configuración -----------------------------------------------------------

# Al menos tres activos financieros (requisito 2 del enunciado).
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "SPY")

PERIOD = "5y"        # ventana histórica a descargar
INTERVAL = "1d"      # frecuencia diaria

# Rutas: se resuelven relativas a la raíz del repositorio.
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"

# Semilla fija para que el fallback sintético sea reproducible byte a byte.
_RANDOM_SEED = 20240501


# --- Descarga con yfinance ---------------------------------------------------

def _download_with_yfinance(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Descarga OHLCV de un símbolo usando yfinance. Puede fallar sin internet."""
    import yfinance as yf  # import diferido para no exigir la librería si hay caché

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"yfinance devolvió un dataframe vacío para {symbol}")

    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "date"})
    keep = ["date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


# --- Fallback sintético determinista -----------------------------------------

def _synthetic_series(symbol: str, period: str) -> pd.DataFrame:
    """
    Genera una serie OHLCV plausible mediante un movimiento browniano
    geométrico. Es determinista por símbolo (semilla derivada del nombre),
    de modo que el dataset generado offline siempre es idéntico.
    """
    n_days = {"1y": 252, "2y": 504, "5y": 1260}.get(period, 1260)

    # Semilla estable por símbolo => reproducibilidad total.
    seed = _RANDOM_SEED + (sum(ord(c) for c in symbol) % 1000)
    rng = np.random.default_rng(seed)

    mu, sigma = 0.0004, 0.012            # deriva y volatilidad diarias
    returns = rng.normal(mu, sigma, n_days)
    price = 100.0 * np.exp(np.cumsum(returns))

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    close = pd.Series(price, index=dates)
    intraday = np.abs(rng.normal(0, sigma / 2, n_days))

    df = pd.DataFrame({
        "date": dates,
        "open": close.shift(1).fillna(close.iloc[0]).to_numpy(),
        "high": close.to_numpy() * (1 + intraday),
        "low": close.to_numpy() * (1 - intraday),
        "close": close.to_numpy(),
        "volume": rng.integers(1_000_000, 50_000_000, n_days),
    })
    logger.warning("Usando datos SINTÉTICOS para %s (yfinance no disponible).", symbol)
    return df


# --- API pública del módulo --------------------------------------------------

def get_symbol_data(
    symbol: str,
    period: str = PERIOD,
    interval: str = INTERVAL,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Devuelve el OHLCV de un símbolo aplicando la cascada
    caché -> yfinance -> sintético, y persiste el resultado en data/raw/.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"{symbol}.csv"

    if use_cache and not refresh and cache_path.exists():
        logger.info("Cargando %s desde caché local: %s", symbol, cache_path)
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df

    try:
        df = _download_with_yfinance(symbol, period, interval)
        logger.info("Descargado %s con yfinance (%d filas).", symbol, len(df))
    except Exception as exc:  # noqa: BLE001 - fallback intencional y controlado
        logger.warning("Fallo yfinance para %s (%s). Se usa fallback.", symbol, exc)
        df = _synthetic_series(symbol, period)

    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    return df


def build_raw_dataset(
    symbols: tuple[str, ...] = SYMBOLS,
    period: str = PERIOD,
    interval: str = INTERVAL,
    refresh: bool = False,
) -> pd.DataFrame:
    """Construye el dataset crudo consolidado (una fila por símbolo y fecha)."""
    frames = []
    for symbol in symbols:
        df = get_symbol_data(symbol, period, interval, refresh=refresh)
        df.insert(0, "symbol", symbol)
        frames.append(df)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sort_values(["symbol", "date"]).reset_index(drop=True)
    return dataset


def main() -> None:
    """Punto de entrada CLI: `python -m financial_api.data`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    dataset = build_raw_dataset()
    print(f"Dataset crudo: {len(dataset)} filas, {dataset['symbol'].nunique()} activos.")
    print(dataset.groupby("symbol")["date"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
