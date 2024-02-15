from argparse import ArgumentParser


def get_options() -> ArgumentParser:
    """
    Парсинг аргументов CLI.

    Returns
    -------
    ArgumentParser
        Объект парсера.
    """
    parser = ArgumentParser()
    # TODO добавить help
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument(
        "--model-type", type=str, required=True, choices=["pretrained", "finetuned"]
    )
    parser.add_argument("--dataset", type=str, required=True)
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
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment", type=str, default="SRGB Inference")
    parser.add_argument("--mlflow-run", type=str, default=None)
    parser.add_argument("--mlflow-system-metrics", action="store_true")
    return parser
