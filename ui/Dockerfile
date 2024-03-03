FROM python:3.10-slim

ARG API_URL

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.6.1 \
    POETRY_NO_INTERACTION=1 \
    PATH="${PATH}:/root/.local/bin" \
    API_URL_ENV=$API_URL

COPY ui /frontend/ui
COPY pyproject.toml poetry.lock /frontend/

WORKDIR /frontend

RUN apt update && \
    apt install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    poetry config virtualenvs.create false && \
    poetry install --with ui --no-root && \
    rm -rf ~/.cache/pypoetry/artifacts && \
    rm -rf ~/.cache/pypoetry/cache

WORKDIR /frontend/ui
EXPOSE 8501

CMD ["sh", "-c", "poetry run streamlit run --server.address 0.0.0.0 --server.port 8501 main.py ${API_URL_ENV}"]
