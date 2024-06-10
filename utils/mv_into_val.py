import os
from argparse import ArgumentParser

import numpy as np


def move(folder: str, test_size: float) -> None:
    """
    Splits dataset into training and testing sets.

    Parameters
    ----------
    folder : str
        Path to dataset folder, which contains 4 subfolders: 270p, 360p, 540p, 1080p.
    test_size : float
        Size of test set [0, 1].

    Returns
    -------
    None
    """
    names = np.array(os.listdir(f"{folder}/270p"))
    val_names = np.random.choice(names, size=int(test_size * len(names)), replace=False)

    os.makedirs(f"{folder}_val", exist_ok=True)
    os.makedirs(f"{folder}_val/270p", exist_ok=True)
    os.makedirs(f"{folder}_val/360p", exist_ok=True)
    os.makedirs(f"{folder}_val/540p", exist_ok=True)
    os.makedirs(f"{folder}_val/1080p", exist_ok=True)

    for name in val_names:
        os.system(f"mv {folder}/270p/{name} {folder}_val/270p/{name}")
        os.system(f"mv {folder}/360p/{name} {folder}_val/360p/{name}")
        os.system(f"mv {folder}/540p/{name} {folder}_val/540p/{name}")
        os.system(f"mv {folder}/1080p/{name} {folder}_val/1080p/{name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        required=True,
        help="path to dataset folder",
    )
    parser.add_argument(
        "-ts",
        "--test-size",
        type=float,
        default=0.2,
        help="size of test set",
    )
    args = parser.parse_args()

    move(args.folder, args.test_size)
