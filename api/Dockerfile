FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.6.1 \
    POETRY_NO_INTERACTION=1 \
    PATH="${PATH}:/root/.local/bin"

COPY api /sr/api
COPY configs /sr/configs
COPY dvc_data /sr/dvc_data
COPY model /sr/model
COPY submodules /sr/submodules
COPY utils /sr/utils
COPY pyproject.toml poetry.lock /sr/

WORKDIR /sr

RUN apt update && apt install -y curl
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN poetry install --with api

WORKDIR /sr/api
EXPOSE 8000

CMD poetry run uvicorn main:app --host 0.0.0.0 --port 8000
