from pathlib import PurePath
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from basicsr.utils import img2tensor
from emt.models.ir_model import IRModel
from tritonclient import http as httpclient


class EMTModel:
    """
    Класс EMT модели.

    Attributes
    ----------
    backend : str
        Backend для запуска модели [torch / onnx / triton].
    nbits : int
        Количество бит данных, взято у авторов модели.
    net_g : optional, Any
        Класс torch.nn.Module модели, доступен только при backend == 'torch'.
    device : optional, Any
        Устройство для запуска модели, доступен только при backend == 'torch'.
    ort_session : optional, onnxruntime.InferenceSession
        Onnx inference сессия, доступна только при backend == 'onnx'.
    ort_input_name : optional, str
        Название входных данных для onnx сессии, доступно только при backend == 'onnx'.
    ort_output_name : optional, str
        Название выходных данных для onnx сессии, доступно только при backend == 'onnx'.
    triton_client : optional, tritonclient.http.InferenceServerClient
        Triton inference сессия, доступна только при backend == 'triton'.
    triton_model_name : optional, str
        Название модели внутри tritonserver, доступно только при backend == 'triton'.
    triton_model_version : optional, str
        Версия модели внутри tritonserver, доступно только при backend == 'triton'.

    Methods
    -------
    enhance(img)
        Повышение качества изображения моделью EMT.
    """

    def __init__(self, root: PurePath, config: dict[str, Any]) -> None:
        """
        Конфигурация класса модели EMT.

        Parameters
        ----------
        root : PurePath
            Путь к корню проекта.
        config : dict[str, Any]
            Конфигурационный словарь модели.

        Returns
        -------
        None
        """
        self.backend = config["backend"]
        self.nbits = config["bit"]
        if self.backend == "torch":
            config["path"]["pretrain_network_g"] = str(
                root / config["path"]["pretrain_network_g"]
            )
            model = IRModel(config)
            self.net_g, self.device, self.nbits = model.net_g, model.device, model.bit
            self.net_g.eval()
            for param in self.net_g.parameters():
                param.requires_grad = False
        elif self.backend == "onnx":
            self.ort_session = ort.InferenceSession(
                root / config["onnx"],
                providers=[
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
            self.ort_input_name = self.ort_session.get_inputs()[0].name
            self.ort_output_name = self.ort_session.get_outputs()[0].name
        elif self.backend == "triton":
            self.triton_client = httpclient.InferenceServerClient(
                url=config["triton_url"]
            )
            self.triton_model_name = config["triton_model_name"]
            self.triton_model_version = config["triton_model_version"]
        else:
            raise ValueError(f"The {self.backend} backend isn't supported")

    @torch.no_grad()
    def enhance(self, img: np.ndarray) -> np.ndarray:
        """
        Повышение качества изображения моделью EMT.

        Parameters
        ----------
        img : np.ndarray
            Изображение в формате (h, w, c).

        Returns
        -------
        np.ndarray
            Изображение в высоком качестве в формате (h, w, c).
        """
        if self.backend == "torch":
            inp_tensor = img2tensor(img).unsqueeze(0).to(self.device)
            out_img = (
                self.net_g(inp_tensor)
                .squeeze(0)
                .detach()
                .cpu()
                .clamp(0, 2**self.nbits - 1.0)
                .round()
                .numpy()
                .transpose(1, 2, 0)
            ).astype(np.uint8)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        elif self.backend == "onnx":
            img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
            out_img = self.ort_session.run(
                [self.ort_output_name],
                {self.ort_input_name: img},
            )[0]
            out_img = (
                out_img.clip(0, 2**self.nbits - 1.0).round().transpose(1, 2, 0)
            ).astype(np.uint8)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        elif self.backend == "triton":
            img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
            inputs = [httpclient.InferInput("lr", img.shape, "FP32")]
            inputs[0].set_data_from_numpy(img, binary_data=True)

            outputs = [httpclient.InferRequestedOutput("hr", binary_data=True)]
            out_img = self.triton_client.infer(
                model_name=self.triton_model_name,
                model_version=str(self.triton_model_version),
                inputs=inputs,
                outputs=outputs,
            ).as_numpy("hr")[0]

            out_img = (
                out_img.clip(0, 2**self.nbits - 1.0).round().transpose(1, 2, 0)
            ).astype(np.uint8)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"The {self.backend} backend isn't supported")
        return out_img


def configure(root: PurePath, config: dict[str, Any]) -> Any:
    """
    Создание модели EMT из конфигурационного словаря.

    Parameters
    ----------
    root : PurePath
        Путь к корню проекта.
    config : dict[str, Any]
        Словарь с конфигурационными параметрами.

    Returns
    -------
    Any
        Объект модели EMT.
    """
    upsampler = EMTModel(root, config)
    return upsampler


def predict(img: np.ndarray, upsampler: Any) -> np.ndarray:
    """
    Перевод LR изображения в HR изображение с использованием EMT.

    Parameters
    ----------
    img : np.ndarray
        Изображение в формате (h, w, c).
    upsampler : Any
        Объект модели EMT.

    Returns
    -------
    np.ndarray
        HR изображение в формате (h*outscale, w*outscale, c).
    """
    out_img = upsampler.enhance(img)
    return out_img
