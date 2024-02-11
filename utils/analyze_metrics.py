import pandas as pd


def main() -> None:
    """
    Функция для визуализации посчитанных метрик.

    Returns
    -------
    None
    """
    # TODO вынести в argparse
    metrics = pd.read_csv("../dvc_data/metrics.csv")

    # TODO вынести в argparse
    show_columns = [
        "time",
        "sr_model",
        "project_type",
        "project_name",
        "lr",
        "hr",
        "split",
        "psnr",
        "ssim",
        "lpips",
    ]

    # TODO сделать возможность писать более общие вопросы
    query = 'project_name == "CSGO" and (split == "train" or split == "val")'

    if query:
        filtered_metrics = metrics.query(query)[show_columns]
    else:
        filtered_metrics = metrics[show_columns]

    print(filtered_metrics)


if __name__ == "__main__":
    # TODO добавить argparse
    main()
