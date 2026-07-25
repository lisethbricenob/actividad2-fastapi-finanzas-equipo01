"""Contratos Pydantic de entrada y salida para la API financiera."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ALLOWED_SYMBOLS = {"AAPL", "MSFT", "SPY"}


class PredictionRequest(BaseModel):
    """Datos requeridos para solicitar una predicción."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "AAPL",
                "prediction_horizon": 1,
                "use_cached_data": True,
            }
        }
    )

    symbol: str = Field(
        ...,
        description="Símbolo financiero utilizado para la inferencia.",
        min_length=1,
        max_length=10,
    )
    prediction_horizon: int = Field(
        default=1,
        ge=1,
        le=1,
        description="Horizonte de predicción en días. El modelo actual usa un día.",
    )
    use_cached_data: bool = Field(
        default=True,
        description="Indica si se debe utilizar la copia local de los datos.",
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        """Normaliza y valida el símbolo recibido."""

        normalized = value.strip().upper()

        if normalized not in ALLOWED_SYMBOLS:
            allowed = ", ".join(sorted(ALLOWED_SYMBOLS))
            raise ValueError(
                f"Símbolo no soportado. Valores permitidos: {allowed}."
            )

        return normalized


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""

    status: Literal["ok", "degraded"]
    model_available: bool
    model_version: str | None = None


class MarketDataResponse(BaseModel):
    """Últimos datos de mercado y variables procesadas de un activo."""

    symbol: str
    date: str
    close: float
    return_1d: float
    ma_5: float
    ma_10: float
    ma_20: float
    volatility_5: float
    volatility_10: float


class PredictionResponse(BaseModel):
    """Respuesta generada por el modelo de clasificación."""

    symbol: str
    prediction: Literal["up", "down"]
    probability_up: float = Field(ge=0.0, le=1.0)
    model_version: str
    prediction_horizon: str
    data_source: Literal["cache", "download", "synthetic", "local"]


class ModelMetadataResponse(BaseModel):
    """Metadatos almacenados junto con el modelo."""

    model_version: str
    model_type: str
    task: str
    training_date: str
    symbols: list[str]
    features: list[str]
    target: str
    prediction_horizon: str
    primary_metric: str
    metrics: dict[str, float]
    n_train: int
    n_test: int


class ErrorResponse(BaseModel):
    """Contrato estándar para respuestas de error."""

    detail: str