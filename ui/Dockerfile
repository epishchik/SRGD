FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.6.1 \
    POETRY_NO_INTERACTION=1 \
    PATH="${PATH}:/root/.local/bin"

COPY ui /frontend/ui
COPY pyproject.toml poetry.lock /frontend/

WORKDIR /frontend

RUN apt update && apt install -y curl
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN poetry install --with ui

WORKDIR /frontend/ui
EXPOSE 8501

CMD poetry run streamlit run main.py
