import json
import logging
import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path, PurePath

import cv2
import numpy as np
import torch
from metric_class import MetricSR

import datasets
import mlflow

sys.path.insert(0, str(Path(__file__).parents[1]))
import model as model_registry
from utils.parse import parse_yaml


def calculate_metrics(
    root: PurePath,
    model_config: str,
    model_type: str,
    dataset_name: str,
    dataset_type: str,
    split: str,
    lr: str,
    hr: str,
    metrics_device: str,
    backend: str,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    mlflow_run: str,
    mlflow_system_metrics: bool,
    triton_url: str,
    only_time: bool,
) -> None:
    """
    Вычисление метрик.

    Parameters
    ----------
    root : PurePath
        Корень проекта.
    model_config : str
        Название конфигурационного файла модели.
    model_type : str
        Тип модели [pretrained / finetuned].
    dataset_name : str
        Название датасета.
    dataset_type : str
        Тип датасета [game_engine / downscale].
    split : str
        Сплит датасета [train / val].
    lr : str
        Низкое разрешение [r270p / r360p / r540p / r1080p].
    hr : str
        Высокое разрешение [r270p / r360p / r540p / r1080p].
    metrics_device : str
        Устройство на котором будет производиться вычисление метрик.
    backend : str
        Бэкенд для модели, перезаписывает настройку из конфига модели [torch / onnx].
    mlflow_tracking_uri : str
        Адрес трекера MLFlow.
    mlflow_experiment : str
        Название эксперимента MLFlow.
    mlflow_run : str
        Название запуска MLFlow.
    mlflow_system_metrics : bool
        Логирование системных метрик.
    triton_url : str
        URL к Triton Inference Server.
    only_time : bool
        Замерить только время inference как метрику.

    Returns
    -------
    None
    """
    map_metric_names = {
        "psnr": "psnr",
        "ssim": "ssim",
        "ms_ssim": "multi_scale_ssim",
        "iw_ssim": "information_weighted_ssim",
        "vifp": "vif_p",
        "fsim": "fsim",
        "srsim": "srsim",
        "gmsd": "gmsd",
        "ms_gmsd": "multi_scale_gmsd",
        "vsi": "vsi",
        "dss": "dss",
        "haarpsi": "haarpsi",
        "mdsi": "mdsi",
        "lpips": "lpips",
        "dists": "dists",
    }
    map_no_ref_metric_names = {"brisque": "brisque", "tv": "total_variation"}

    for metric_name, map_metric_name in map_no_ref_metric_names.items():
        map_metric_names[metric_name] = map_metric_name
    metric_names = [k for k in map_metric_names.keys()]
    no_ref_metric_names = [k for k in map_no_ref_metric_names.keys()]

    model_config_path = root / f"configs/model/{model_type}/{model_config}.yaml"
    metric_config_path = root / f"configs/metric/{dataset_type}/{dataset_name}.yaml"

    model_config_dct = parse_yaml(str(model_config_path))
    metric_config_dct = parse_yaml(str(metric_config_path))

    save_dir = root / metric_config_dct["output_path"]
    os.makedirs(save_dir, exist_ok=True)
    project_type, project_name = metric_config_dct["project_name"].split("_")
    save_path = (
        save_dir / f"{model_config}_{model_type}_{dataset_name}"
        f"_{dataset_type}_{split}_{lr}_{hr}.json"
    )

    if backend:
        model_config_dct["backend"] = backend

    if triton_url:
        model_config_dct["triton_url"] = triton_url

    upsampler = getattr(model_registry, model_config_dct["model"]).configure(
        root, model_config_dct
    )

    if split == "train":
        dataset_split = datasets.Split.TRAIN
    elif split == "val":
        dataset_split = datasets.Split.VALIDATION
    else:
        raise ValueError(f"{split} does not exist")

    dataset = datasets.load_dataset(
        metric_config_dct["repository"],
        name=metric_config_dct["project_name"],
        split=dataset_split,
        streaming=True,
    )

    if not only_time:
        metric_calculator = MetricSR(
            metric_names,
            no_ref_metric_names,
            map_metric_names,
            device=metrics_device,
        )

    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")

    logs_path = root / "logs"
    os.makedirs(logs_path, exist_ok=True)

    file_handler = logging.FileHandler(logs_path / "metric.log")

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_file_handler = logging.StreamHandler(stream=sys.stdout)
    stream_file_handler.setFormatter(formatter)
    logger.addHandler(stream_file_handler)

    logger.setLevel("INFO")

    logger.info(f"model = {model_config}")
    logger.info(f"project type = {project_type}")
    logger.info(f"project name = {project_name}")
    logger.info(f"low resolution = {lr}")
    logger.info(f"high resolution = {hr}")
    logger.info(f"split = {split}")
    logger.info(f"save_path = {save_path}")

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        if not mlflow.get_experiment_by_name(mlflow_experiment):
            mlflow.create_experiment(mlflow_experiment)
        mlflow.set_experiment(mlflow_experiment)

        mlflow.start_run(run_name=mlflow_run, log_system_metrics=mlflow_system_metrics)

        mlflow.log_params({f"model_{k}": v for k, v in model_config_dct.items()})
        mlflow.log_params({f"metric_{k}": v for k, v in metric_config_dct.items()})
        mlflow.log_params(
            {
                "low resolution": lr,
                "high resolution": hr,
                "split": split,
                "model config path": model_config_path,
                "metric config path": metric_config_path,
            }
        )

    time_lst = []
    for idx, el in enumerate(dataset):
        rgb_lr = np.asarray(el[lr].convert("RGB"), dtype=np.float32)
        bgr_lr = cv2.cvtColor(rgb_lr, cv2.COLOR_RGB2BGR)

        rgb_hr = np.asarray(el[hr].convert("RGB"), dtype=np.float32)
        bgr_hr = cv2.cvtColor(rgb_hr, cv2.COLOR_RGB2BGR)

        start_time = time.time()
        res_hr = getattr(model_registry, model_config_dct["model"]).predict(
            bgr_lr, upsampler
        )
        total_time = time.time() - start_time
        time_lst += [total_time]

        res_hr, bgr_hr = torch.from_numpy(res_hr), torch.from_numpy(bgr_hr)
        res_hr = res_hr.unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        bgr_hr = bgr_hr.unsqueeze(0).permute(0, 3, 1, 2) / 255.0

        metric_str = [f"image = {idx + 1},"]
        if not only_time:
            metric_calculator.calculate(res_hr, bgr_hr)
            for metric_name in metric_names:
                metric_val = metric_calculator.metric_history[metric_name][-1]
                metric_str += [f"{metric_name} = {metric_val:.3f},"]
                if mlflow_tracking_uri:
                    mlflow.log_metric(metric_name, metric_val, step=idx)

        if mlflow_tracking_uri:
            mlflow.log_metric("time", total_time, step=idx)

        metric_str += [f"time = {total_time:.3f}"]
        metric_str = " ".join(metric_str)
        logger.info(metric_str)

    json_dct = {
        "cli_args": {
            "model_config": model_config,
            "model_type": model_type,
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            "split": split,
            "lr": lr,
            "hr": hr,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_experiment": mlflow_experiment,
            "mlflow_run": mlflow_run,
            "mlflow_system_metrics": mlflow_system_metrics,
        },
        "model_config": model_config_dct,
        "metric_config": metric_config_dct,
        "metrics": {},
    }

    if not only_time:
        for metric_name in metric_names:
            metric_val = metric_calculator.calculate_total(metric_name)
            json_dct["metrics"][metric_name] = float(f"{metric_val:.4f}")
            if mlflow_tracking_uri:
                mlflow.log_metric(f"mean {metric_name}", metric_val)

    mean_time = sum(time_lst[1:]) / (len(time_lst) - 1)  # first run is always slower
    if mlflow_tracking_uri:
        mlflow.log_metric("mean time", mean_time)
        mlflow.end_run()
    json_dct["time"] = float(f"{mean_time:.4f}")
    json_dct["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(save_path, "w") as f:
        json.dump(json_dct, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    # TODO добавить help
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument(
        "--model-type", type=str, required=True, choices=["pretrained", "finetuned"]
    )
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument(
        "--dataset-type", type=str, required=True, choices=["downscale", "game_engine"]
    )
    parser.add_argument("--split", type=str, required=True, choices=["train", "val"])
    parser.add_argument(
        "--lr", type=str, required=True, choices=["r270p", "r360p", "r540p", "r1080p"]
    )
    parser.add_argument(
        "--hr", type=str, required=True, choices=["r270p", "r360p", "r540p", "r1080p"]
    )
    parser.add_argument("--metrics-device", type=str, default="cuda")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment", type=str, default="SRGB Inference")
    parser.add_argument("--mlflow-run", type=str, default=None)
    parser.add_argument("--mlflow-system-metrics", action="store_true")
    parser.add_argument("--triton-url", type=str, default=None)
    parser.add_argument("--only-time", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parents[1]
    calculate_metrics(
        root,
        args.model_config,
        args.model_type,
        args.dataset_name,
        args.dataset_type,
        args.split,
        args.lr,
        args.hr,
        args.metrics_device,
        args.backend,
        args.mlflow_tracking_uri,
        args.mlflow_experiment,
        args.mlflow_run,
        args.mlflow_system_metrics,
        args.triton_url,
        args.only_time,
    )
