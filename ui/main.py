import io
from typing import Any

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError
from requests import Response

MODELNAME2API = {
    ("Real-ESRGAN", "pretrained", "x4"): "pretrained/RealESRGAN_x4plus",
    ("Real-ESRGAN", "pretrained", "x2"): "pretrained/RealESRGAN_x2plus",
    ("Real-ESRGAN", "finetuned", "x4"): None,
    ("Real-ESRGAN", "finetuned", "x2"): None,
    ("ResShift", "pretrained", "x4"): "pretrained/ResShift_RealSRx4",
    ("ResShift", "pretrained", "x2"): None,
    ("ResShift", "finetuned", "x4"): None,
    ("ResShift", "finetuned", "x2"): None,
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

    model_name = st.selectbox("Model name", ("Real-ESRGAN", "ResShift"))
    model_type = st.selectbox("Model type", ("pretrained", "finetuned"))
    upsacle_ratio = st.selectbox("Upscale ratio", ("x4", "x2"))

    configure_model_name = MODELNAME2API[(model_name, model_type, upsacle_ratio)]

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
