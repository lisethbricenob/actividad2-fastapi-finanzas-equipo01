"""Pruebas automatizadas de los endpoints de FastAPI."""

from fastapi.testclient import TestClient

from financial_api.api import app


client = TestClient(app)


def test_root() -> None:
    """Debe retornar información básica de la API."""

    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["message"] == "API financiera educativa"
    assert body["docs"] == "/docs"
    assert body["health"] == "/health"


def test_health() -> None:
    """Debe confirmar que la API y el modelo están disponibles."""

    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["model_available"] is True
    assert body["model_version"] == "random_forest_v1"


def test_market_data_valid_symbol() -> None:
    """Debe devolver datos y variables para un símbolo válido."""

    response = client.get("/market-data/AAPL")

    assert response.status_code == 200

    body = response.json()

    expected_fields = {
        "symbol",
        "date",
        "close",
        "return_1d",
        "ma_5",
        "ma_10",
        "ma_20",
        "volatility_5",
        "volatility_10",
    }

    assert expected_fields.issubset(body.keys())
    assert body["symbol"] == "AAPL"
    assert isinstance(body["date"], str)
    assert isinstance(body["close"], float)
    assert isinstance(body["return_1d"], float)
    assert isinstance(body["ma_5"], float)
    assert isinstance(body["ma_10"], float)
    assert isinstance(body["ma_20"], float)
    assert isinstance(body["volatility_5"], float)
    assert isinstance(body["volatility_10"], float)


def test_market_data_normalizes_symbol() -> None:
    """Debe aceptar un símbolo escrito en minúsculas."""

    response = client.get("/market-data/msft")

    assert response.status_code == 200
    assert response.json()["symbol"] == "MSFT"


def test_market_data_invalid_symbol() -> None:
    """Debe responder 400 para un símbolo no soportado."""

    response = client.get("/market-data/TSLA")

    assert response.status_code == 400

    body = response.json()

    assert "Símbolo no soportado" in body["detail"]
    assert "AAPL, MSFT, SPY" in body["detail"]


def test_predict_valid_request() -> None:
    """Debe generar una predicción para una solicitud válida."""

    payload = {
        "symbol": "AAPL",
        "prediction_horizon": 1,
        "use_cached_data": True,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    expected_fields = {
        "symbol",
        "prediction",
        "probability_up",
        "model_version",
        "prediction_horizon",
        "data_source",
    }

    assert expected_fields.issubset(body.keys())
    assert body["symbol"] == "AAPL"
    assert body["prediction"] in {"up", "down"}
    assert 0.0 <= body["probability_up"] <= 1.0
    assert body["model_version"] == "random_forest_v1"
    assert body["prediction_horizon"] == "next_day"
    assert body["data_source"] in {
        "cache",
        "download",
        "synthetic",
        "local",
    }


def test_predict_normalizes_symbol() -> None:
    """Debe normalizar un símbolo válido escrito en minúsculas."""

    payload = {
        "symbol": "spy",
        "prediction_horizon": 1,
        "use_cached_data": True,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["symbol"] == "SPY"


def test_predict_invalid_symbol() -> None:
    """Pydantic debe rechazar símbolos no soportados."""

    payload = {
        "symbol": "TESLA",
        "prediction_horizon": 1,
        "use_cached_data": True,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    body = response.json()

    assert "detail" in body
    assert "Símbolo no soportado" in str(body["detail"])


def test_predict_invalid_horizon() -> None:
    """Debe rechazar horizontes diferentes de un día."""

    payload = {
        "symbol": "AAPL",
        "prediction_horizon": 2,
        "use_cached_data": True,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_model_metadata() -> None:
    """Debe devolver los metadatos reales del modelo."""

    response = client.get("/model/metadata")

    assert response.status_code == 200

    body = response.json()

    expected_fields = {
        "model_version",
        "model_type",
        "task",
        "training_date",
        "symbols",
        "features",
        "target",
        "prediction_horizon",
        "primary_metric",
        "metrics",
        "n_train",
        "n_test",
    }

    assert expected_fields.issubset(body.keys())
    assert body["model_version"] == "random_forest_v1"
    assert body["model_type"] == "RandomForestClassifier"
    assert body["task"] == "trend_classification"
    assert body["prediction_horizon"] == "next_day"
    assert set(body["symbols"]) == {"AAPL", "MSFT", "SPY"}
    assert len(body["features"]) == 12
    assert body["n_train"] > 0
    assert body["n_test"] > 0