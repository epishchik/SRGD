import os
from argparse import ArgumentParser

import numpy as np


def main() -> None:
    """
    Отделяет тестовую часть заданного размера в отдельную директорию.

    Returns
    -------
    None
    """
    parser = ArgumentParser()
    parser.add_argument("--folder", type=str)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    names = np.array(os.listdir(f"{args.folder}/270p"))

    val_names = np.random.choice(
        names, size=int(args.test_size * len(names)), replace=False
    )

    os.makedirs(f"{args.folder}_val", exist_ok=True)
    os.makedirs(f"{args.folder}_val/270p", exist_ok=True)
    os.makedirs(f"{args.folder}_val/360p", exist_ok=True)
    os.makedirs(f"{args.folder}_val/540p", exist_ok=True)
    os.makedirs(f"{args.folder}_val/1080p", exist_ok=True)

    for name in val_names:
        os.system(f"mv {args.folder}/270p/{name} {args.folder}_val/270p/{name}")
        os.system(f"mv {args.folder}/360p/{name} {args.folder}_val/360p/{name}")
        os.system(f"mv {args.folder}/540p/{name} {args.folder}_val/540p/{name}")
        os.system(f"mv {args.folder}/1080p/{name} {args.folder}_val/1080p/{name}")


if __name__ == "__main__":
    main()
