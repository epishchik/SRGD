import sys
from pathlib import Path, PurePath
from typing import Any

import numpy as np

project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

# TODO обернуть в python module, добавить в pyproject.toml и requirements.txt
from submodules.resshift.inference_resshift import get_configs_from_global_config
from submodules.resshift.sampler import ResShiftSampler


def configure(root: PurePath, config: dict[str, Any]) -> Any:
    """
    Функция создания модели  из конфигурационного словаря.

    Parameters
    ----------
    root : PurePath
        Путь к корню проекта.
    config : dict[str, Any]
        Словарь с конфигурационными параметрами.

    Returns
    -------
    Any
        Объект модели ResShift.
    """
    chop_size = config["chop_size"]
    seed = config["seed"]
    fp32 = config["fp32"]
    device = config["device"]

    configs, chop_stride, desired_min_size = get_configs_from_global_config(
        root, config
    )

    resshift_sampler = ResShiftSampler(
        configs,
        chop_size=chop_size,
        chop_stride=chop_stride,
        chop_bs=1,
        use_fp16=not fp32,
        seed=seed,
        desired_min_size=desired_min_size,
        device=device,
        package_root=config["resshift_location"],
    )

    return resshift_sampler


def predict(
    img: np.ndarray,
    upsampler: Any,
) -> np.ndarray:
    """
    Перевод LR изображения в HR изображение с использованием ResShift.

    Parameters
    ----------
    img : np.ndarray
        Изображение в формате (h, w, c).
    upsampler : Any
        Объект модели ResShift.

    Returns
    -------
    np.ndarray
        HR изображение в формате (h*outscale, w*outscale, c).
    """
    out_img = upsampler.inference_single(img)
    return out_img
