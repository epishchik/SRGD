import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import torch.cuda
import yaml
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

root = Path(__file__).parents[1]
sys.path.insert(0, str(root))

import model as model_registry
from utils.parse import parse_yaml


def load_triton_model(triton_url: str, model_name: str) -> None:
    """
    Load specific model into tritonserver.

    Parameters
    ----------
    triton_url : str
        Triton inference server URL.
    model_name : str
        Model name form triton model_repository.

    Returns
    -------
    None
    """
    requests.post(f"http://{triton_url}/v2/repository/models/{model_name}/load")


def unload_triton_model(triton_url: str, model_name: str) -> None:
    """
    Unload specific model from tritonserver.

    Parameters
    ----------
    triton_url : str
        Triton inference server URL.
    model_name : str
        Model name form triton model_repository.

    Returns
    -------
    None
    """
    requests.post(f"http://{triton_url}/v2/repository/models/{model_name}/unload")


def set_cpu_mode(config: dict[str, Any]) -> dict[str, Any]:
    """
    Set CPU mode.

    Parameters
    ----------
    config : dict[str, Any]
        Model configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Changed model configuration dictionary.
    """
    if config["model"] == "emt_model":
        config["num_gpu"] = 0
    elif config["model"] == "real_esrgan":
        config["gpu_id"] = -1
    elif config["model"] == "resshift":
        config["device"] = "cpu"
    else:
        raise ValueError(f"{config['model']} model isn't supported.")
    return config


app = FastAPI()

init_model = "RealESRGAN_x4plus"
app.config_path = str(root / f"configs/model/pretrained/{init_model}.yaml")
app.config = parse_yaml(app.config_path)

if app.config["backend"] == "triton":
    app.triton_url = "triton:8000"
    app.config["triton_url"] = app.triton_url
    load_triton_model(app.triton_url, init_model)

if torch.cuda.device_count() < 1 and app.config["backend"] != "triton":
    app.config = set_cpu_mode(app.config)

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
    Get file with short overview of project.

    Returns
    -------
    FileResponse
        Text file.
    """
    info_file = FileResponse(
        path="./extra/info.txt", filename="info.txt", media_type="text"
    )
    logger.debug("used /info")
    return info_file


@app.post("/configure_model/name")
def configure_model(config_name: str) -> dict[str, Any]:
    """
    Configure model using its config name.

    Parameters
    ----------
    config_name : str
        Model config name in pretrained/model or trained/model format.

    Returns
    -------
    dict[str, Any]
        Dictionary with model configuration parameters.
    """
    if "backend" in app.config and app.config["backend"] == "triton":
        try:
            tmp_model_name = app.config["triton_model_name"]
            unload_triton_model(app.triton_url, tmp_model_name)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="something went wrong with unloading triton model.",
            )

    previous_config, previous_config_path = app.config, app.config_path
    try:
        app.config_path = str(root / f"configs/model/{config_name}.yaml")
        app.config = parse_yaml(app.config_path)
    except ValueError:
        app.config_path = previous_config_path
        app.config = previous_config
        raise HTTPException(
            status_code=400, detail=f"config '{config_name}' is not supported."
        )

    if "backend" in app.config and app.config["backend"] == "triton":
        try:
            app.config["triton_url"] = app.triton_url
            load_triton_model(app.triton_url, app.config["triton_model_name"])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="something went wrong with loading triton model.",
            )

    try:
        if torch.cuda.device_count() < 1 and app.config["backend"] != "triton":
            app.config = set_cpu_mode(app.config)
        app.upsampler = getattr(model_registry, app.config["model"]).configure(
            root, app.config
        )
    except AttributeError:
        raise HTTPException(
            status_code=400, detail=f"model '{app.config['model']}' is not supported."
        )
    except ValueError:
        raise HTTPException(
            status_code=400, detail="something went wrong with model configuration."
        )

    logger.debug("used /configure_model/name")
    logger.debug(f"set new config path = {app.config_path}")
    logger.debug(f"set new config dict = {app.config}")

    return app.config


@app.post("/configure_model/file")
def configure_model_file(config_file: UploadFile) -> dict[str, Any]:
    """
    Configure model using its config YAML file.

    Parameters
    ----------
    config_file : UploadFile
        Config YAML file.

    Returns
    -------
    dict[str, Any]
        Dictionary with model configuration parameters.
    """
    if config_file.content_type not in ["application/x-yaml"]:
        raise HTTPException(
            status_code=400,
            detail=f"file type of {config_file.content_type} is not supported",
        )

    if "backend" in app.config and app.config["backend"] == "triton":
        try:
            tmp_model_name = app.config["triton_model_name"]
            unload_triton_model(app.triton_url, tmp_model_name)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="something went wrong with unloading triton model.",
            )

    app.config_path = None
    app.config = yaml.safe_load(config_file.file.read())
    app.config["filename"] = Path(config_file.filename).stem

    if "backend" in app.config and app.config["backend"] == "triton":
        try:
            app.config["triton_url"] = app.triton_url
            load_triton_model(app.triton_url, app.config["triton_model_name"])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="something went wrong with loading triton model.",
            )

    try:
        if torch.cuda.device_count() < 1 and app.config["backend"] != "triton":
            app.config = set_cpu_mode(app.config)
        app.upsampler = getattr(model_registry, app.config["model"]).configure(
            root, app.config
        )
    except AttributeError:
        raise HTTPException(
            status_code=400, detail=f"model {app.config['model']} is not supported."
        )
    except ValueError:
        raise HTTPException(
            status_code=400, detail="something went wrong with model configuration."
        )

    logger.debug("used /configure_model/file")
    logger.debug(f"set new config path = {app.config_path}")
    logger.debug(f"set new config dict = {app.config}")

    return app.config


def _upscale(img: np.ndarray) -> tuple[bytes, tuple[int, int], tuple[int, int], float]:
    """
    Enhance image.

    Parameters
    ----------
    img : np.ndarray
        np.ndarray image in (h, w, c) format.

    Returns
    -------
    tuple[bytes, tuple[int, int], tuple[int, int], float]
        0. Encoded image,
        1. (low resolution height, low resolution width),
        2. (high resolution height, high resolution width),
        3. Inference time.
    """
    try:
        h, w = img.shape[0], img.shape[1]
    except AttributeError:
        raise HTTPException(
            status_code=400,
            detail="incorrect img object, should be 3 dimensional numpy.ndarray.",
        )

    if "outscale" in app.config:
        outscale = app.config["outscale"]
    elif "network_g" in app.config and "upscale" in app.config["network_g"]:
        outscale = app.config["network_g"]["upscale"]
    else:
        outscale = 4

    h_up, w_up = int(h * outscale), int(w * outscale)
    if h_up > 4320 or w_up > 7680:
        raise HTTPException(
            status_code=400,
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
    Upscale example image using configured model.

    Returns
    -------
    Response
        Generated HR image.
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
    Upscale image from file using configured model.

    Parameters
    ----------
    image_file : UploadFile
        File with image in (h, w, c) format.

    Returns
    -------
    Response
        Generated HR image.
    """
    if image_file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=400,
            detail=f"file type of {image_file.content_type} is not supported",
        )

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
