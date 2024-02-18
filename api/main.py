import logging
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

root = Path(__file__).parents[1]
sys.path.insert(0, str(root))

from model.emt_model import configure as configure_emt
from model.emt_model import predict as predict_emt
from model.real_esrgan import configure as configure_real_esrgan
from model.real_esrgan import predict as predict_real_esrgan
from model.resshift import configure as configure_resshift
from model.resshift import predict as predict_resshift
from utils.parse import parse_yaml

app = FastAPI()

app.config_path = str(root / "configs/model/pretrained/RealESRGAN_x4plus.yaml")
app.config = parse_yaml(app.config_path)

app.upsampler = configure_real_esrgan(root, app.config)

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


def _configure_model() -> None:
    """
    Конфигурация super-resolution модели по названию.

    Returns
    -------
    None
    """
    if app.config["model"] == "real_esrgan":
        app.upsampler = configure_real_esrgan(root, app.config)
    elif app.config["model"] == "resshift":
        app.upsampler = configure_resshift(root, app.config)
    elif app.config["model"] == "emt":
        app.upsampler, app.emt_device, app.emt_nbits = configure_emt(root, app.config)
    else:
        raise ValueError(f"{app.config['model']} incorrect model type.")


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
    app.config_path = str(root / f"configs/model/{config_name}.yaml")
    app.config = parse_yaml(app.config_path)

    _configure_model()

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
    app.config_path = None
    app.config = yaml.safe_load(config_file.file.read())
    app.config["filename"] = Path(config_file.filename).stem

    _configure_model()

    logger.debug("used /configure_model/file")
    logger.debug(f"set new config path = {app.config_path}")
    logger.debug(f"set new config dict = {app.config}")

    return app.config


def _upscale(img: np.ndarray) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """
    Функция повышения качества изображения.

    Parameters
    ----------
    img : np.ndarray
        Изображение в формате np.ndarray размера (h, w, c).

    Returns
    -------
    tuple[bytes, tuple[int, int], tuple[int, int]]
        Закодированное в байткод изображение высокого качества,
        (low resolution height, low resolution width),
        (high resolution height, high resolution width).
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

    if app.config["model"] == "real_esrgan":
        out_img = predict_real_esrgan(img, app.upsampler, outscale)
    elif app.config["model"] == "resshift":
        out_img = predict_resshift(img, app.upsampler)
    elif app.config["model"] == "emt":
        out_img = predict_emt(img, app.upsampler, app.emt_device, app.emt_nbits)
    else:
        raise ValueError(f"{app.config['model']} incorrect model type.")

    _, enc_img = cv2.imencode(".png", out_img)
    bytes_img = enc_img.tobytes()
    return bytes_img, (h, w), (h_up, w_up)


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
    bytes_img, (h, w), (h_up, w_up) = _upscale(img)

    logger.debug("used /upscale/example")
    logger.debug(f"low_res shape = ({h},{w}), ups_res shape = ({h_up},{w_up})")
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
    bytes_img, (h, w), (h_up, w_up) = _upscale(img)

    logger.debug("used /upscale/file")
    logger.debug(f"low_res shape = ({h},{w}), ups_res shape = ({h_up},{w_up})")
    return Response(bytes_img, media_type="image/png")
