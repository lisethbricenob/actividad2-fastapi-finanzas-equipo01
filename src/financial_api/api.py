"""API financiera educativa construida con FastAPI."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Path

from financial_api import data as data_module
from financial_api import features as features_module
from financial_api.predict import (
    PredictionServiceError,
    load_metadata,
    load_model,
    predict_symbol,
)
from financial_api.schemas import (
    ALLOWED_SYMBOLS,
    ErrorResponse,
    HealthResponse,
    MarketDataResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
)


app = FastAPI(
    title="API Financiera Educativa",
    description=(
        "Servicio académico para consultar datos de mercado y generar "
        "predicciones de tendencia. No constituye asesoría financiera."
    ),
    version="0.1.0",
)


@app.get(
    "/",
    tags=["General"],
)
def root() -> dict[str, str]:
    """Devuelve información básica del servicio."""

    return {
        "message": "API financiera educativa",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={
        503: {"model": ErrorResponse},
    },
    tags=["Estado"],
)
def health() -> HealthResponse:
    """Verifica que la API y el modelo estén disponibles."""

    try:
        model = load_model()
        metadata = load_metadata()

        return HealthResponse(
            status="ok",
            model_available=model is not None,
            model_version=metadata.get("model_version"),
        )

    except PredictionServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@app.get(
    "/market-data/{symbol}",
    response_model=MarketDataResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Mercado"],
)
def market_data(
    symbol: str = Path(
        ...,
        description="Símbolo financiero: AAPL, MSFT o SPY.",
        min_length=1,
        max_length=10,
    ),
) -> MarketDataResponse:
    """Devuelve los datos recientes y las variables principales de un activo."""

    normalized_symbol = symbol.strip().upper()

    if normalized_symbol not in ALLOWED_SYMBOLS:
        allowed = ", ".join(sorted(ALLOWED_SYMBOLS))
        raise HTTPException(
            status_code=400,
            detail=f"Símbolo no soportado. Valores permitidos: {allowed}.",
        )

    try:
        raw_data = data_module.get_symbol_data(
            symbol=normalized_symbol,
            use_cache=True,
        )

        dataset = raw_data.copy()
        dataset.insert(0, "symbol", normalized_symbol)

        processed = features_module.build_features(dataset)

        if processed.empty:
            raise HTTPException(
                status_code=500,
                detail=(
                    "No existen suficientes datos para calcular "
                    f"las variables de {normalized_symbol}."
                ),
            )

        # Última fila válida de variables procesadas.
        latest = processed.sort_values("date").iloc[-1]

        # Se toma el precio de cierre correspondiente exactamente
        # a la misma fecha de la fila de variables procesadas.
        latest_date = latest["date"]

        matching_close = raw_data.loc[
            raw_data["date"] == latest_date,
            "close",
        ]

        if matching_close.empty:
            raise HTTPException(
                status_code=500,
                detail=(
                    "No fue posible encontrar el precio de cierre "
                    f"correspondiente a la fecha {latest_date}."
                ),
            )

        latest_close = matching_close.iloc[0]

        return MarketDataResponse(
            symbol=normalized_symbol,
            date=latest_date.isoformat(),
            close=round(float(latest_close), 4),
            return_1d=round(float(latest["return_1d"]), 6),
            ma_5=round(float(latest["ma_5"]), 4),
            ma_10=round(float(latest["ma_10"]), 4),
            ma_20=round(float(latest["ma_20"]), 4),
            volatility_5=round(float(latest["volatility_5"]), 6),
            volatility_10=round(float(latest["volatility_10"]), 6),
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"No fue posible obtener los datos de mercado: {exc}",
        ) from exc


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Predicción"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Genera una predicción de tendencia para el siguiente día."""

    try:
        result = predict_symbol(
            symbol=request.symbol,
            use_cached_data=request.use_cached_data,
        )

        return PredictionResponse(**result)

    except PredictionServiceError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error inesperado durante la predicción: {exc}",
        ) from exc


@app.get(
    "/model/metadata",
    response_model=ModelMetadataResponse,
    responses={
        503: {"model": ErrorResponse},
    },
    tags=["Modelo"],
)
def model_metadata() -> ModelMetadataResponse:
    """Devuelve los metadatos del modelo entrenado."""

    try:
        metadata = load_metadata()
        return ModelMetadataResponse(**metadata)

    except PredictionServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc