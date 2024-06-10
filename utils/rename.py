import os
from argparse import ArgumentParser


def rename(root: str) -> None:
    """
    Recursively rename all files in subdirectories of root directory,
    files with the same name will be renamed with the same name.

    Parameters
    ----------
    root : str
        Path to root directory with all subdirectories.

    Returns
    -------
    None
    """
    folders = [os.path.join(root, folder) for folder in os.listdir(root)]

    files = os.listdir(folders[0])
    map_names = {k: str(v + 1).zfill(5) for v, k in enumerate(files)}

    for i, folder in enumerate(folders):
        for src_name, dst_name in map_names.items():
            ext = src_name.split(".")[-1]
            src_name = os.path.join(folder, src_name)
            dst_name = os.path.join(folder, f"{dst_name}.{ext}")
            os.rename(src_name, dst_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        required=True,
        help="path to root directory with all subdirectories",
    )
    args = parser.parse_args()

    rename(args.root)
