from argparse import ArgumentParser

import numpy as np
import pandas as pd
from rle import decode


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    filename_column_name: str,
    rle_column_name: str,
) -> float:
    """
    Вычисление метрики PSNR для лидерборда на kaggle.

    Parameters
    ----------
    solution : pd.DataFrame
        Датафрейм содержащий закодированные таргет изображения.
    submission : pd.DataFrame
        Датафрейм содержащий закодированные полученные изображения.
    row_id_column_name : str
        Название первой колонки, которая используется как id для join.
    filename_column_name : str
        Название колонки, в которой хранятся названия файлов.
    rle_column_name : str
        Название колонки, в которой хранятся закодированные изображения.

    Returns
    -------
    float
        Значение метрики PSNR.
    """
    if row_id_column_name:
        del solution[row_id_column_name]
        del submission[row_id_column_name]
    del solution[filename_column_name]
    del submission[filename_column_name]

    real_rle, pred_rle = (
        solution[rle_column_name].values,
        submission[rle_column_name].values,
    )
    maxi = 20 * np.log10(255)
    max_psnr = 100
    cnt, psnr, eps = 0, 0.0, 1e-6
    for real, pred in zip(real_rle, pred_rle):
        real_img, pred_img = decode(eval(real)), decode(eval(pred))
        inf_psnr = maxi - 10 * np.log10(np.mean((real_img - pred_img) ** 2) + eps)
        psnr += inf_psnr if inf_psnr <= max_psnr else max_psnr
        cnt += 1
    return psnr / cnt


if __name__ == "__main__":
    parser = ArgumentParser()
    # TODO добавить help
    parser.add_argument("--submission", type=str, default="submission.csv")
    parser.add_argument("--solution", type=str, default="solution.csv")
    parser.add_argument("--col1", type=str, default=None)
    parser.add_argument("--col2", type=str, default="filename")
    parser.add_argument("--col3", type=str, default="rle")
    args = parser.parse_args()

    psnr = score(
        pd.read_csv(args.solution).copy(),
        pd.read_csv(args.submission).copy(),
        args.col1,
        args.col2,
        args.col3,
    )

    print(f"PSNR = {psnr:.3f}")
