from pathlib import PurePath
from typing import Any

import cv2
import numpy as np
from basicsr.utils import img2tensor
from emt.models.ir_model import IRModel


def configure(root: PurePath, config: dict[str, Any]) -> Any:
    """
    Функция создания модели EMT из конфигурационного словаря.

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
    config["path"]["pretrain_network_g"] = str(
        root / config["path"]["pretrain_network_g"]
    )

    # TODO выяснить правильный способ загрузки
    model = IRModel(config)
    net_g, device, nbits = model.net_g, model.device, model.bit
    net_g.eval()
    for param in net_g.parameters():
        param.requires_grad = False

    return net_g, device, nbits


def predict(img: np.ndarray, upsampler: Any, device: Any, nbits: int) -> np.ndarray:
    """
    Перевод LR изображения в HR изображение с использованием EMT.

    Parameters
    ----------
    img : np.ndarray
        Изображение в формате (h, w, c).
    upsampler : Any
        Объект модели EMT.
    device : Any
        Устройство на котором будут выполняться вычисления.
    nbits : int
        Количество бит для кодирования каждого пикселя.

    Returns
    -------
    np.ndarray
        HR изображение в формате (h*outscale, w*outscale, c).
    """
    # TODO правильно сделать preprocessing и postprocessing
    inp_tensor = img2tensor(img).unsqueeze(0).to(device)
    out_img = (
        upsampler(inp_tensor)
        .squeeze(0)
        .detach()
        .cpu()
        .clamp(0, 2**nbits - 1.0)
        .round()
        .numpy()
        .transpose(1, 2, 0)
    ).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    return out_img
