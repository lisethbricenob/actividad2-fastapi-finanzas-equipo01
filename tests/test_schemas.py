"""Pruebas de los contratos Pydantic de la API financiera."""

import pytest
from pydantic import ValidationError

from financial_api.schemas import PredictionRequest


def test_prediction_request_valid_symbol() -> None:
    """Debe aceptar y normalizar un símbolo válido."""

    request = PredictionRequest(
        symbol="aapl",
        prediction_horizon=1,
        use_cached_data=True,
    )

    assert request.symbol == "AAPL"
    assert request.prediction_horizon == 1
    assert request.use_cached_data is True


def test_prediction_request_trims_spaces() -> None:
    """Debe eliminar espacios y normalizar el símbolo."""

    request = PredictionRequest(symbol="  msft  ")

    assert request.symbol == "MSFT"


def test_prediction_request_rejects_invalid_symbol() -> None:
    """Debe rechazar símbolos que no fueron usados por el modelo."""

    with pytest.raises(ValidationError) as error:
        PredictionRequest(symbol="TESLA")

    message = str(error.value)

    assert "Símbolo no soportado" in message
    assert "AAPL, MSFT, SPY" in message


def test_prediction_request_rejects_invalid_horizon() -> None:
    """El modelo actual solo permite horizonte de un día."""

    with pytest.raises(ValidationError):
        PredictionRequest(
            symbol="AAPL",
            prediction_horizon=2,
        )


def test_prediction_request_default_values() -> None:
    """Debe asignar correctamente los valores predeterminados."""

    request = PredictionRequest(symbol="SPY")

    assert request.symbol == "SPY"
    assert request.prediction_horizon == 1
    assert request.use_cached_data is True