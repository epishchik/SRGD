import os.path
import sys
from argparse import ArgumentParser
from pathlib import Path, PurePath

import cv2
import pandas as pd
from rle import encode
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
import model as model_registry
from utils.parse import parse_yaml


def prediction(
    root: PurePath,
    sample_submission: str,
    lr_folder: str,
    output_file: str,
    model_config: str,
    model_type: str,
) -> None:
    """
    Get prediction for sample submission using local model.

    Parameters
    ----------
    root : str
        Path to project root directory.
    sample_submission : str
        Path to sample submission file.
    lr_folder : str
        Path to test dataset folder with LR images.
    output_file : str
        Path to output file with predictions.
    model_config : str
        Model config filename.
    model_type : str
        [pretrained | finetuned].

    Returns
    -------
    None
    """
    model_config_path = root / f"configs/model/{model_type}/{model_config}.yaml"
    model_config_dct = parse_yaml(str(model_config_path))

    upsampler = getattr(model_registry, model_config_dct["model"]).configure(
        root, model_config_dct
    )

    submission_df = pd.read_csv(sample_submission)

    filenames = submission_df["filename"].values
    for i in tqdm(range(len(filenames))):
        filename = filenames[i]
        init_img = cv2.imread(os.path.join(lr_folder, filename))
        out_img = getattr(model_registry, model_config_dct["model"]).predict(
            init_img, upsampler
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
        "-c",
        "--model-config",
        type=str,
        default="RealESRGAN_x4plus",
        help="model config filename",
    )
    parser.add_argument(
        "-t",
        "--model-type",
        type=str,
        default="pretrained",
        help="[pretrained | finetuned]",
    )
    args = parser.parse_args()

    root = Path(__file__).parents[1]

    prediction(
        root,
        args.sample_submission,
        args.lr_folder,
        args.output_file,
        args.model_config,
        args.model_type,
    )
