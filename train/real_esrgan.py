from pathlib import Path

from basicsr.train import train_pipeline_hf


def train() -> None:
    root = Path(__file__).parents[1]
    train_pipeline_hf(root)


if __name__ == "__main__":
    train()
