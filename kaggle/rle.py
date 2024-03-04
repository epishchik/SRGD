import os
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def encode(img: np.ndarray) -> str:
    img_to_encode = img.astype(np.uint8)
    img_to_encode = img_to_encode.flatten()
    img_to_encode = np.append(img_to_encode, -1)

    cnt, rle = 1, ""
    for i in range(1, img_to_encode.shape[0]):
        if img_to_encode[i] == img_to_encode[i - 1]:
            cnt += 1
        else:
            rle += f"{img_to_encode[i - 1]} {cnt} "
            cnt = 1
    return rle[:-1]


def decode(encoded_img: str) -> np.ndarray:
    rle = list(map(int, encoded_img.split()))
    decoded_img = []
    for i in range(0, len(rle), 2):
        decoded_img += [rle[i]] * rle[i + 1]
    return np.array(decoded_img, dtype=np.uint8)


def encode_folder(folder: str, save_path: str = "solution.csv") -> None:
    dct = {"filename": [], "rle": []}

    files = os.listdir(folder)
    for i in tqdm(range(len(files))):
        img = cv2.imread(os.path.join(folder, files[i]))
        h, w, c = img.shape
        encoded_img = encode(img)
        decoded_img = decode(encoded_img).reshape(h, w, c)
        assert (img - decoded_img).sum() == 0, "encoded != decoded"
        dct["filename"] += [files[i]]
        dct["rle"] += [encode(img)]

    df = pd.DataFrame(dct)
    df.to_csv(save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    # TODO добавить help
    parser.add_argument("-f", "--folder", type=str, required=True)
    parser.add_argument("-s", "--save-path", type=str, default="solution.csv")
    args = parser.parse_args()

    encode_folder(args.folder, save_path=args.save_path)
