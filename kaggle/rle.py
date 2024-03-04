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
    Кодирование изображения без потерь для submission на платформе kaggle.
    Kaggle не поддерживает отправки кроме как через submission.csv.
    Аналогичным образом необходимо хранить таргет в solution.csv.

    Parameters
    ----------
    img : np.ndarray
        cv2.imread(f) - BGR изображение в формате (h, w, c), c = 3 (даже для .png).

    Returns
    -------
    bytes
        Закодированное изображение.
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
    Декодирование изображения - обратная операция к кодированию функцией encode.

    Parameters
    ----------
    encoded_img : bytes
        Закодированное изображение.

    Returns
    -------
    np.ndarray
        BGR изображение в формате (h, w, c), c = 3 (даже для .png).
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
    Кодирование изображений из папки (таргет изображения из тестового датасета).
    Сохранение в подготовленный к загрузке на kaggle solution.csv.

    Parameters
    ----------
    folder : str
        Путь к папке с таргет изображениями.
    save_path : str
        Путь к создаваемому файлу solution.csv.
    public_size : float
        Размер public части тестового датасета для лидерборда.

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
    # TODO добавить help
    parser.add_argument("-f", "--folder", type=str, required=True)
    parser.add_argument("-s", "--save-path", type=str, default="solution.csv")
    parser.add_argument("-p", "--public-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    encode_folder(args.folder, save_path=args.save_path, public_size=args.public_size)
