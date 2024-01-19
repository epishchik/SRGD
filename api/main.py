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


@app.post("/configure_model/name")
def configure_model(config_name: str) -> dict[str, Any]:
    """
    Конфигурация super-resolution модели по названию.

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

    if app.config["model"] == "real_esrgan":
        app.upsampler = configure_real_esrgan(root, app.config)
    elif app.config["model"] == "resshift":
        app.upsampler = configure_resshift(root, app.config)
    else:
        raise ValueError(f"{app.config['model']} incorrect method name.")

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

    if app.config["model"] == "real_esrgan":
        app.upsampler = configure_real_esrgan(root, app.config)
    elif app.config["model"] == "resshift":
        app.upsampler = configure_resshift(root, app.config)
    else:
        raise ValueError(f"{app.config['model']} incorrect method name.")

    logger.debug("used /configure_model/file")
    logger.debug(f"set new config path = {app.config_path}")
    logger.debug(f"set new config dict = {app.config}")

    return app.config


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

    h, w = img.shape[0], img.shape[1]
    outscale = app.config["outscale"]

    h_up, w_up = int(h * outscale), int(w * outscale)
    if h_up > 4320 or w_up > 7680:
        raise HTTPException(
            status_code=481,
            detail=f"potential output resolution ({h_up, w_up}) is too large, "
            "max possible resolution is (4320, 7680) pixels in (h, w) format.",
        )

    if app.config["model"] == "real_esrgan":
        out_img = predict_real_esrgan(
            img, app.upsampler, outscale, app.config["use_face_enhancer"]
        )
    elif app.config["model"] == "resshift":
        out_img = predict_resshift(img, app.upsampler)
    else:
        raise ValueError(f"{app.config['model']} incorrect method name.")

    _, enc_img = cv2.imencode(".png", out_img)
    bytes_img = enc_img.tobytes()

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

    h, w = img.shape[0], img.shape[1]
    outscale = app.config["outscale"]

    h_up, w_up = int(h * outscale), int(w * outscale)
    if h_up > 4320 or w_up > 7680:
        raise HTTPException(
            status_code=481,
            detail=f"potential output resolution ({h_up, w_up}) is too large, "
            "max possible resolution is (4320, 7680) pixels in (h, w) format.",
        )

    if app.config["model"] == "real_esrgan":
        out_img = predict_real_esrgan(
            img, app.upsampler, outscale, app.config["use_face_enhancer"]
        )
    elif app.config["model"] == "resshift":
        out_img = predict_resshift(img, app.upsampler)
    else:
        raise ValueError(f"{app.config['model']} incorrect method name.")

    _, enc_img = cv2.imencode(".png", out_img)
    bytes_img = enc_img.tobytes()

    logger.debug("used /upscale/file")
    logger.debug(f"low_res shape = ({h},{w}), ups_res shape = ({h_up},{w_up})")
    return Response(bytes_img, media_type="image/png")
