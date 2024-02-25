import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import yaml
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

root = Path(__file__).parents[1]
sys.path.insert(0, str(root))

import model as model_registry
from utils.parse import parse_yaml


def load_triton_model(triton_url: str, model_name: str) -> None:
    """
    Загрузка модели на GPU из репозитория. Необходимая мера, чтобы не превысить объем
    VRAM на GPU, потому что моделей много и по умолчанию они все загружаются в VRAM.

    Parameters
    ----------
    triton_url : str
        URL к tritonserver, например triton:8000 для запуска через docker compose.
    model_name : str
        Название модели из triton model_repository.

    Returns
    -------
    None
    """
    requests.post(f"http://{triton_url}/v2/repository/models/{model_name}/load")


def unload_triton_model(triton_url: str, model_name: str) -> None:
    """
    Выгрузка модели из GPU. Необходимая мера, чтобы не превысить объем VRAM на GPU,
    потому что моделей много и по умолчанию они все загружаются в VRAM.

    Parameters
    ----------
    triton_url : str
        URL к tritonserver, например triton:8000 для запуска через docker compose.
    model_name : str
        Название модели из triton model_repository.

    Returns
    -------
    None
    """
    requests.post(f"http://{triton_url}/v2/repository/models/{model_name}/unload")


app = FastAPI()

init_model = "RealESRGAN_x4plus"
app.config_path = str(root / f"configs/model/pretrained/{init_model}.yaml")
app.config = parse_yaml(app.config_path)

app.triton_url = "triton:8000"
app.config["triton_url"] = app.triton_url

load_triton_model(app.triton_url, init_model)
app.upsampler = getattr(model_registry, app.config["model"]).configure(root, app.config)

logger = logging.getLogger("uvicorn")
formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")

logs_path = str(root / "logs")
os.makedirs(logs_path, exist_ok=True)

file_handler = logging.FileHandler(f"{logs_path}/api.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.setLevel("DEBUG")

logger.debug(f"set config path = {app.config_path}")
logger.debug(f"set config dict = {app.config}")


@app.get("/info")
def info() -> FileResponse:
    """
    Получение файла с краткой информацией о проекте.

    Returns
    -------
    FileResponse
        Текстовый файл.
    """
    info_file = FileResponse(
        path="./extra/info.txt", filename="info.txt", media_type="text"
    )
    logger.debug("used /info")
    return info_file


@app.post("/configure_model/name")
def configure_model(config_name: str) -> dict[str, Any]:
    """
    Конфигурация super-resolution модели по названию конфига.

    Parameters
    ----------
    config_name : str
        Название модели в формате pretrained/model или trained/model.

    Returns
    -------
    dict[str, Any]
        Словарь конфигурационных параметров.
    """
    if "triton_model_name" in app.config:
        tmp_model_name = app.config["triton_model_name"]
        unload_triton_model(app.triton_url, tmp_model_name)

    app.config_path = str(root / f"configs/model/{config_name}.yaml")
    app.config = parse_yaml(app.config_path)
    app.config["triton_url"] = app.triton_url

    if "triton_model_name" in app.config:
        load_triton_model(app.triton_url, app.config["triton_model_name"])
    app.upsampler = getattr(model_registry, app.config["model"]).configure(
        root, app.config
    )

    logger.debug("used /configure_model/name")
    logger.debug(f"set new config path = {app.config_path}")
    logger.debug(f"set new config dict = {app.config}")

    return app.config


@app.post("/configure_model/file")
def configure_model_file(config_file: UploadFile) -> dict[str, Any]:
    """
    Конфигурация super-resolution модели конфигурационным yaml файлом.

    Parameters
    ----------
    config_file : UploadFile
        Конфигурационный .yaml файл.

    Returns
    -------
    dict[str, Any]
        Словарь конфигурационных параметров.
    """
    if "triton_model_name" in app.config:
        tmp_model_name = app.config["triton_model_name"]
        unload_triton_model(app.triton_url, tmp_model_name)

    app.config_path = None
    app.config = yaml.safe_load(config_file.file.read())
    app.config["filename"] = Path(config_file.filename).stem
    app.config["triton_url"] = app.triton_url

    if "triton_model_name" in app.config:
        load_triton_model(app.triton_url, app.config["triton_model_name"])
    app.upsampler = getattr(model_registry, app.config["model"]).configure(
        root, app.config
    )

    logger.debug("used /configure_model/file")
    logger.debug(f"set new config path = {app.config_path}")
    logger.debug(f"set new config dict = {app.config}")

    return app.config


def _upscale(img: np.ndarray) -> tuple[bytes, tuple[int, int], tuple[int, int], float]:
    """
    Функция повышения качества изображения.

    Parameters
    ----------
    img : np.ndarray
        Изображение в формате np.ndarray размера (h, w, c).

    Returns
    -------
    tuple[bytes, tuple[int, int], tuple[int, int], float]
        Закодированное в байткод изображение высокого качества,
        (low resolution height, low resolution width),
        (high resolution height, high resolution width),
        время предсказания на inference.
    """
    h, w = img.shape[0], img.shape[1]

    if "outscale" in app.config:
        outscale = app.config["outscale"]
    elif "network_g" in app.config and "upscale" in app.config["network_g"]:
        outscale = app.config["network_g"]["upscale"]
    else:
        raise ValueError("there is no 'outscale' parameter.")

    h_up, w_up = int(h * outscale), int(w * outscale)
    if h_up > 4320 or w_up > 7680:
        raise HTTPException(
            status_code=481,
            detail=f"potential output resolution ({h_up, w_up}) is too large, "
            "max possible resolution is (4320, 7680) pixels in (h, w) format.",
        )

    start_time = time.time()
    out_img = getattr(model_registry, app.config["model"]).predict(img, app.upsampler)
    total_time = time.time() - start_time

    _, enc_img = cv2.imencode(".png", out_img)
    bytes_img = enc_img.tobytes()
    return bytes_img, (h, w), (h_up, w_up), total_time


@app.post("/upscale/example")
def upscale_example() -> Response:
    """
    Пример работы super-resolution модели на заготовленном LR изображении.

    Returns
    -------
    Response
        Сгенерированное HR изображение.
    """
    img = cv2.imread("./extra/example.png", cv2.IMREAD_UNCHANGED)
    bytes_img, (h, w), (h_up, w_up), total_time = _upscale(img)

    logger.debug("used /upscale/example")
    logger.debug(
        f"low_res shape = ({h},{w}), "
        f"ups_res shape = ({h_up},{w_up}), "
        f"ups_time = {total_time:.3f}"
    )
    return Response(bytes_img, media_type="image/png")


@app.post("/upscale/file")
def upscale(image_file: UploadFile) -> Response:
    """
    Обработка super-resolution моделью изображения из файла.

    Parameters
    ----------
    image_file : UploadFile
        Файл с LR изображением.

    Returns
    -------
    Response
        Сгенерированное HR изображение.
    """
    raw = np.fromstring(image_file.file.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    bytes_img, (h, w), (h_up, w_up), total_time = _upscale(img)

    logger.debug("used /upscale/file")
    logger.debug(
        f"low_res shape = ({h},{w}), "
        f"ups_res shape = ({h_up},{w_up}), "
        f"ups_time = {total_time:.3f}"
    )
    return Response(bytes_img, media_type="image/png")
