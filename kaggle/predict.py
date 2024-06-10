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
    Get prediction for sample submission using huggingface model.

    Parameters
    ----------
    sample_submission : str
        Path to sample submission file.
    lr_folder : str
        Path to test dataset folder with LR images.
    output_file : str
        Path to output file with predictions.
    model : str
        Name (repository) of huggingface model.
    device : str
        Device on which the model will be run.
    simple_resize : str
        If specified then the upscaling will be done using deterministic interpolation.

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
    parser.add_argument(
        "-s",
        "--sample-submission",
        type=str,
        required=True,
        help="path to sample submission file",
    )
    parser.add_argument(
        "-l",
        "--lr-folder",
        type=str,
        required=True,
        help="path to test dataset folder with LR images",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="submission.csv",
        help="path to output file with predictions",
    )
    parser.add_argument(
        "-r",
        "--simple-resize",
        type=str,
        default=None,
        help="interpolation method for upscaling or None for model upscaling",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="epishchik/RealESRNet_x4plus",
        help="name of huggingface model",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda:0",
        help="device on which the model will be run",
    )
    args = parser.parse_args()

    prediction(
        args.sample_submission,
        args.lr_folder,
        args.output_file,
        args.model,
        args.device,
        args.simple_resize,
    )
