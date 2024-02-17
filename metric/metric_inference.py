import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from metric_class import MetricSR
from options import get_options

import datasets
import mlflow


def main() -> None:
    """
    Вычисление, запись в файл метрик metrics.csv, логирование в MLFlow.
    В качестве датасета используется один из проектов из
    epishchik/super-resolution-games с HuggingFace.

    Returns
    -------
    None
    """
    root = Path(__file__).parents[1]
    sys.path.insert(0, str(root))

    from model.emt_model import configure as configure_emt
    from model.emt_model import predict as predict_emt
    from model.real_esrgan import configure as configure_real_esrgan
    from model.real_esrgan import predict as predict_real_esrgan
    from model.resshift import configure as configure_resshift
    from model.resshift import predict as predict_resshift
    from utils.parse import parse_yaml

    metric_file_header = [
        "time",
        "sr_model",
        "project_type",
        "project_name",
        "lr",
        "hr",
        "split",
    ]

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
    metric_file_header += metric_names

    args = get_options().parse_args()
    model_name = args.model_config

    model_config_path = root / f"configs/model/{args.model_type}/{model_name}.yaml"

    metric_config_path = (
        root / f"configs/metric/{args.dataset_type}/{args.dataset}.yaml"
    )

    model_config = parse_yaml(str(model_config_path))
    metric_config = parse_yaml(str(metric_config_path))

    save_dir = root / metric_config["output_path"]
    os.makedirs(save_dir, exist_ok=True)
    project_type, project_name = metric_config["project_name"].split("_")
    save_path = save_dir / "metrics.csv"

    if not os.path.exists(save_path):
        with open(save_path, "w") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(metric_file_header)

    if model_config["model"] == "real_esrgan":
        upsampler = configure_real_esrgan(root, model_config)
    elif model_config["model"] == "resshift":
        upsampler = configure_resshift(root, model_config)
    elif model_config["model"] == "emt":
        upsampler, emt_device, emt_nbits = configure_emt(root, model_config)
    else:
        raise ValueError(f"{model_config['model']} incorrect model type.")

    split_config = args.split
    if split_config == "train":
        split = datasets.Split.TRAIN
    elif split_config == "val":
        split = datasets.Split.VALIDATION
    else:
        raise ValueError(f"{split_config} does not exist")

    dataset = datasets.load_dataset(
        metric_config["repository"],
        name=metric_config["project_name"],
        split=split,
        streaming=True,
    )

    lr, hr = args.lr, args.hr
    metric_calculator = MetricSR(metric_names, no_ref_metric_names, map_metric_names)

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

    logger.info(f"model = {model_name}")
    logger.info(f"project type = {project_type}")
    logger.info(f"project name = {project_name}")
    logger.info(f"low resolution = {lr}")
    logger.info(f"high resolution = {hr}")
    logger.info(f"split = {split_config}")

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        if not mlflow.get_experiment_by_name(args.mlflow_experiment):
            mlflow.create_experiment(args.mlflow_experiment)
        mlflow.set_experiment(args.mlflow_experiment)

        mlflow.start_run(
            run_name=args.mlflow_run, log_system_metrics=args.mlflow_system_metrics
        )

        mlflow.log_params({f"model_{k}": v for k, v in model_config.items()})
        mlflow.log_params({f"metric_{k}": v for k, v in metric_config.items()})
        mlflow.log_params(
            {
                "low resolution": lr,
                "high resolution": hr,
                "split": split,
                "model config path": model_config_path,
                "metric config path": metric_config_path,
            }
        )

    for idx, el in enumerate(dataset):
        rgb_lr = np.asarray(el[lr].convert("RGB"), dtype=np.float32)
        bgr_lr = cv2.cvtColor(rgb_lr, cv2.COLOR_RGB2BGR)

        rgb_hr = np.asarray(el[hr].convert("RGB"), dtype=np.float32)
        bgr_hr = cv2.cvtColor(rgb_hr, cv2.COLOR_RGB2BGR)

        outscale = int(bgr_hr.shape[0] / bgr_lr.shape[0])
        if model_config["model"] == "real_esrgan":
            res_hr = predict_real_esrgan(
                bgr_lr, upsampler, outscale, model_config["use_face_enhancer"]
            )
        elif model_config["model"] == "resshift":
            res_hr = predict_resshift(bgr_lr, upsampler)
        elif model_config["model"] == "emt":
            res_hr = predict_emt(bgr_lr, upsampler, emt_device, emt_nbits)
        else:
            raise ValueError(f"{model_config['model']} incorrect model type.")

        res_hr, bgr_hr = torch.from_numpy(res_hr), torch.from_numpy(bgr_hr)
        res_hr = res_hr.unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        bgr_hr = bgr_hr.unsqueeze(0).permute(0, 3, 1, 2) / 255.0

        if torch.cuda.is_available():
            res_hr = res_hr.cuda()
            bgr_hr = bgr_hr.cuda()

        metric_calculator.calculate(res_hr, bgr_hr)

        metric_str = f"image = {idx+1}, "
        for metric_name in metric_names:
            metric_val = metric_calculator.metric_history[metric_name][-1]
            metric_str += f"{metric_name} = {metric_val:.3f}, "
            if args.mlflow_tracking_uri:
                mlflow.log_metric(metric_name, metric_val, step=idx)
        logger.info(metric_str[:-2])

    metrics_total = []
    for metric_name in metric_names:
        metric_val = metric_calculator.calculate_total(metric_name)
        metrics_total += [f"{metric_val:.3f}"]
        if args.mlflow_tracking_uri:
            mlflow.log_metric(f"mean {metric_name}", metric_val)
    if args.mlflow_tracking_uri:
        mlflow.end_run()

    with open(save_path, "a") as csv_file:
        csv_writer = csv.writer(csv_file)
        time = datetime.now()
        csv_writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                model_name,
                project_type,
                project_name,
                lr[1:],
                hr[1:],
                split_config,
                *metrics_total,
            ]
        )


if __name__ == "__main__":
    main()
