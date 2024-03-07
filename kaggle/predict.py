import os.path
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
from basicsr.utils.img_util import img2tensor, tensor2img
from rle import encode
from tqdm import tqdm
from transformers import AutoModel


def prediction(
    sample_submission: str,
    lr_folder: str,
    output_file: str,
    model: str,
    device: str,
    simple_resize: str,
) -> None:
    """
    Получение предсказания с помощью предобученной модели с huggingface.

    Parameters
    ----------
    sample_submission : str
        Путь к файлу с примером структуры посылки на kaggle.
    lr_folder : str
        Путь к папке тестового датасета с изображениями в низком качестве.
    output_file : str
        Путь для сохранения, полученного файла с предсказаниями.
    model : str
        Название (repository) модели на huggingface.
    device : str
        Устройство на котором будут выполняться вычисления.
    simple_resize : str
        Если указан, то upscale делается указанным алгоритмом, а не моделью.

    Returns
    -------
    None
    """
    model = AutoModel.from_pretrained(model, trust_remote_code=True).to(device)
    submission_df = pd.read_csv(sample_submission)

    filenames = submission_df["filename"].values
    for i in tqdm(range(len(filenames))):
        filename = filenames[i]
        init_img = cv2.imread(os.path.join(lr_folder, filename))
        if simple_resize:
            out_img = cv2.resize(
                init_img,
                (init_img.shape[1] * 4, init_img.shape[0] * 4),
                interpolation=getattr(cv2, simple_resize),
            ).astype(np.uint8)
        else:
            init_tnsr = (
                img2tensor(init_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
            ) / 255.0
            out_tnsr = model(init_tnsr)
            out_img = tensor2img(
                out_tnsr,
                rgb2bgr=True,
                out_type=np.uint8,
            )
        submission_df.loc[i, "rle"] = encode(out_img)
    submission_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    # TODO добавить help
    parser.add_argument("-s", "--sample-submission", type=str, required=True)
    parser.add_argument("-l", "--lr-folder", type=str, required=True)
    parser.add_argument("-o", "--output-file", type=str, default="submission.csv")
    parser.add_argument("-r", "--simple-resize", type=str, default=None)
    parser.add_argument(
        "-m", "--model", type=str, default="epishchik/RealESRNet_x4plus"
    )
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    args = parser.parse_args()

    prediction(
        args.sample_submission,
        args.lr_folder,
        args.output_file,
        args.model,
        args.device,
        args.simple_resize,
    )
