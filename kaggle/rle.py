import base64
import os
import zlib
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def encode(img: np.ndarray) -> bytes:
    """
    Lossless encoding of images for submission on kaggle platform.
    Kaggle doesn't support submissons not in csv format.
    In the same way target images should be stored.

    Parameters
    ----------
    img : np.ndarray
        cv2.imread(f) - BGR image in (h, w, c) format, c = 3 (even for png format).

    Returns
    -------
    bytes
        Encoded image as bytes.
    """
    img_to_encode = img.astype(np.uint8)
    img_to_encode = img_to_encode.flatten()
    img_to_encode = np.append(img_to_encode, -1)

    cnt, rle = 1, []
    for i in range(1, img_to_encode.shape[0]):
        if img_to_encode[i] == img_to_encode[i - 1]:
            cnt += 1
            if cnt > 255:
                rle += [img_to_encode[i - 1], 255]
                cnt = 1
        else:
            rle += [img_to_encode[i - 1], cnt]
            cnt = 1

    compressed = zlib.compress(bytes(rle), zlib.Z_BEST_COMPRESSION)
    base64_bytes = base64.b64encode(compressed)
    return base64_bytes


def decode(encoded_img: bytes) -> np.ndarray:
    """
    Reverse operation for encode function to get original images.

    Parameters
    ----------
    encoded_img : bytes
        Encoded image as bytes.

    Returns
    -------
    np.ndarray
        BGR image in (h, w, c) format, c = 3 (even for png format).
    """
    rle = zlib.decompress(base64.b64decode(encoded_img))
    decoded_img = []
    for i in range(0, len(rle), 2):
        decoded_img += [rle[i]] * rle[i + 1]
    return np.array(decoded_img, dtype=np.uint8)


def encode_folder(
    folder: str, save_path: str = "solution.csv", public_size: float = 0.3
) -> None:
    """
    Encode images from folder (target images from test set) and save into solution.csv.

    Parameters
    ----------
    folder : str
        Path to directory with target images.
    save_path : str
        Path to save solution.csv.
    public_size : float
        Size of public part of the dataset for the leaderboard.

    Returns
    -------
    None
    """
    dct = {"filename": [], "rle": [], "Usage": []}

    files = os.listdir(folder)
    public_ind = np.random.choice(
        [i for i in range(len(files))],
        size=int(public_size * len(files)),
        replace=False,
    )
    for i in tqdm(range(len(files))):
        img = cv2.imread(os.path.join(folder, files[i]))
        h, w, c = img.shape
        encoded_img = encode(img)
        decoded_img = decode(encoded_img).reshape(h, w, c)
        assert (img - decoded_img).sum() == 0, "encoded != decoded"
        dct["filename"] += [files[i]]
        dct["rle"] += [encode(img)]
        if i in public_ind:
            dct["Usage"] += ["Public"]
        else:
            dct["Usage"] += ["Private"]

    df = pd.DataFrame(dct)
    df.to_csv(save_path, index=True, index_label="id")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        required=True,
        help="path to folder with target images",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        default="solution.csv",
        help="path to save solution.csv",
    )
    parser.add_argument(
        "-p",
        "--public-size",
        type=float,
        default=0.3,
        help="size of public part of the dataset for the leaderboard",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed value",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    encode_folder(args.folder, save_path=args.save_path, public_size=args.public_size)
