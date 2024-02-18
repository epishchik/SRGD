import io
from typing import Any

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError
from requests import Response

MODELNAME2API = {
    # Real-ESRGAN
    ("Real-ESRGAN", "Pretrained", "Game Engine", "x4"): "pretrained/RealESRGAN_x4plus",
    ("Real-ESRGAN", "Pretrained", "Game Engine", "x3"): None,
    ("Real-ESRGAN", "Pretrained", "Game Engine", "x2"): "pretrained/RealESRGAN_x2plus",
    ("Real-ESRGAN", "Pretrained", "Downscale", "x4"): "pretrained/RealESRGAN_x4plus",
    ("Real-ESRGAN", "Pretrained", "Downscale", "x3"): None,
    ("Real-ESRGAN", "Pretrained", "Downscale", "x2"): "pretrained/RealESRGAN_x2plus",
    (
        "Real-ESRGAN",
        "Finetuned",
        "Game Engine",
        "x4",
    ): "finetuned/RealESRGAN_x4plus_GameEngineData",
    ("Real-ESRGAN", "Finetuned", "Game Engine", "x3"): None,
    ("Real-ESRGAN", "Finetuned", "Game Engine", "x2"): None,
    ("Real-ESRGAN", "Finetuned", "Downscale", "x4"): None,
    ("Real-ESRGAN", "Finetuned", "Downscale", "x3"): None,
    ("Real-ESRGAN", "Finetuned", "Downscale", "x2"): None,
    # ResShift
    ("ResShift", "Pretrained", "Game Engine", "x4"): "pretrained/ResShift_RealSRx4",
    ("ResShift", "Pretrained", "Game Engine", "x3"): None,
    ("ResShift", "Pretrained", "Game Engine", "x2"): None,
    ("ResShift", "Pretrained", "Downscale", "x4"): "pretrained/ResShift_RealSRx4",
    ("ResShift", "Pretrained", "Downscale", "x3"): None,
    ("ResShift", "Pretrained", "Downscale", "x2"): None,
    ("ResShift", "Finetuned", "Game Engine", "x4"): None,
    ("ResShift", "Finetuned", "Game Engine", "x3"): None,
    ("ResShift", "Finetuned", "Game Engine", "x2"): None,
    ("ResShift", "Finetuned", "Downscale", "x4"): None,
    ("ResShift", "Finetuned", "Downscale", "x3"): None,
    ("ResShift", "Finetuned", "Downscale", "x2"): None,
    # EMT
    ("EMT", "Pretrained", "Game Engine", "x4"): "pretrained/EMT_x4",
    ("EMT", "Pretrained", "Game Engine", "x3"): "pretrained/EMT_x3",
    ("EMT", "Pretrained", "Game Engine", "x2"): "pretrained/EMT_x2",
    ("EMT", "Pretrained", "Downscale", "x4"): "pretrained/EMT_x4",
    ("EMT", "Pretrained", "Downscale", "x3"): "pretrained/EMT_x3",
    ("EMT", "Pretrained", "Downscale", "x2"): "pretrained/EMT_x2",
    ("EMT", "Finetuned", "Game Engine", "x4"): None,
    ("EMT", "Finetuned", "Game Engine", "x3"): None,
    ("EMT", "Finetuned", "Game Engine", "x2"): None,
    ("EMT", "Finetuned", "Downscale", "x4"): None,
    ("EMT", "Finetuned", "Downscale", "x3"): None,
    ("EMT", "Finetuned", "Downscale", "x2"): None,
}


def upscale_file(image: Any, server_url: str) -> Response:
    """
    Повышение качества изображения при помощи запроса к API.

    Parameters
    ----------
    image : Any
        Изображение низкого разрешения в формате (h, w, c).
    server_url : str
        IP-адрес API backend сервиса.

    Returns
    -------
    Response
        Ответ API сервиса на запрос.
    """
    files = [
        (
            "image_file",
            (
                image.name,
                image,
                "image/jpeg",
            ),
        )
    ]

    response = requests.request("POST", server_url + "/upscale/file", files=files)
    return response


def configure_model(config_name: str, server_url: str) -> Response:
    """
    Конфигурация модели при помощи запроса к API.

    Parameters
    ----------
    config_name : str
        Название конфигурационного файла модели.
    server_url : str
        IP-адрес API backend сервиса.

    Returns
    -------
    Response
        Ответ API сервиса на запрос.
    """
    params = {"config_name": config_name}
    response = requests.request(
        "POST", server_url + "/configure_model/name", params=params
    )
    return response


def main() -> None:
    """
    Отображение frontend части.

    Returns
    -------
    None
    """

    base_url = "http://api:8000"
    title = "Super Resolution in Games"

    st.title(title)

    model_name = st.selectbox("Model name", ("Real-ESRGAN", "ResShift", "EMT"))
    model_type = st.selectbox("Model type", ("Pretrained", "Finetuned"))
    upsacle_ratio = st.selectbox("Upscale ratio", ("x4", "x3", "x2"))
    data_type = st.selectbox("Data type", ("Game Engine", "Downscale"))

    configure_model_name = MODELNAME2API[
        (model_name, model_type, data_type, upsacle_ratio)
    ]

    if st.button("Configure model"):
        if configure_model_name:
            response = configure_model(configure_model_name, base_url)
            if response.status_code == 200:
                st.write("Configuration has been applied.")
            else:
                st.write("Something went wrong with the configuration process.")
        else:
            st.write("This configuration isn't available.")

    input_image = st.file_uploader("Insert image")

    if st.button("Upscale"):
        col1, col2 = st.columns(2)

        if input_image:
            try:
                upscaled_bytes = upscale_file(input_image, base_url)
                original_image = Image.open(input_image).convert("RGB")
                upscaled_image = Image.open(io.BytesIO(upscaled_bytes.content)).convert(
                    "RGB"
                )
                _, hr_h = upscaled_image.size

                col1.header("Low Resolution")
                col1.image(original_image, use_column_width=True)

                col2.header("High Resolution")
                col2.image(upscaled_image, use_column_width=True)

                st.download_button(
                    label="Download upscaled image",
                    data=io.BytesIO(upscaled_bytes.content),
                    file_name=f"{input_image.name}_{hr_h}p",
                    mime="image/png",
                )
            except UnidentifiedImageError:
                st.write("Something went wrong with the upscaling process.")
        else:
            st.write("Insert an image!")


if __name__ == "__main__":
    main()
