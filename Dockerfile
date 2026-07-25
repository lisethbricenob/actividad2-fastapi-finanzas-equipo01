FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PORT=8000

WORKDIR /app

RUN pip install --no-cache-dir "poetry>=2.0,<3.0"

COPY pyproject.toml poetry.lock README.md ./

RUN poetry install --no-root --no-ansi \
    && rm -rf "${POETRY_CACHE_DIR}"

COPY src ./src
COPY artifacts ./artifacts
COPY data ./data
COPY reports ./reports
COPY tests ./tests
COPY TEAM.md ./

RUN poetry install --only-root --no-ansi

EXPOSE 8000

CMD ["sh", "-c", "uvicorn financial_api.api:app --host 0.0.0.0 --port ${PORT:-8000}"]