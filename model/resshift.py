import sys
from pathlib import Path, PurePath
from typing import Any

import cv2
import numpy as np

project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

# TODO prepare as a python module, add to pyproject.toml and requirements.txt
from submodules.resshift.inference_resshift import get_configs_from_global_config
from submodules.resshift.sampler import ResShiftSampler


def configure(root: PurePath, config: dict[str, Any]) -> Any:
    """
    Create ResShift model from configuration dictionary.

    Parameters
    ----------
    root : PurePath
        Path to project root directory.
    config : dict[str, Any]
        Dictionary with configuration parameters.

    Returns
    -------
    Any
        ResShift model.
    """
    configs, chop_stride, desired_min_size = get_configs_from_global_config(
        root, config
    )
    resshift_sampler = ResShiftSampler(
        configs,
        chop_size=config["chop_size"],
        chop_stride=chop_stride,
        chop_bs=1,
        use_fp16=not config["fp32"],
        seed=config["seed"],
        desired_min_size=desired_min_size,
        device=config["device"],
        package_root=config["resshift_location"],
    )
    return resshift_sampler


def predict(
    img: np.ndarray,
    upsampler: Any,
) -> np.ndarray:
    """
    Enhance low resolution image using ResShift model.

    Parameters
    ----------
    img : np.ndarray
        Image in (h, w, c) format.
    upsampler : Any
        ResShift model.

    Returns
    -------
    np.ndarray
        High resolution image in (h*, w*, c) format.
    """
    out_img = upsampler.inference_single(img)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    return out_img
