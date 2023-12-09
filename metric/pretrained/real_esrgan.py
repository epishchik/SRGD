import csv
import os
import sys
from datetime import datetime

import cv2
import datasets
import numpy as np
import piq
import torch


def main(root: str):
    sys.path.insert(0, root)
    from model.real_esrgan import configure, predict
    from utils.parse import parse_yaml

    real_esrgan_config_name = 'RealESRGAN_x4plus'
    model_config_path = os.path.join(
        root,
        f'configs/model/{real_esrgan_config_name}.yaml'
    )

    metric_config_path = os.path.join(
        root,
        'configs/metric/game_engine/ActionRPG.yaml'
    )

    model_config = parse_yaml(model_config_path)
    metric_config = parse_yaml(metric_config_path)

    save_dir = os.path.join(root, metric_config['output_path'])
    os.makedirs(save_dir, exist_ok=True)
    project_type, project_name = metric_config['project_name'].split('_')
    save_path = os.path.join(save_dir, 'metrics.csv')

    if not os.path.exists(save_path):
        with open(save_path, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                'time',
                'sr_model',
                'project_type',
                'project_name',
                'lr',
                'hr',
                'split',
                'psnr',
                'ssim'
            ])

    upsampler = configure(root, model_config)

    split_config = metric_config['split']
    if split_config == 'train':
        split = datasets.Split.TRAIN
    elif split_config == 'val':
        split = datasets.Split.VALIDATION
    else:
        raise ValueError(f'{split_config} does not exist')

    dataset = datasets.load_dataset(
        metric_config['repository'],
        name=metric_config['project_name'],
        split=split,
        streaming=True
    )

    lr, hr = metric_config['lr'], metric_config['hr']
    psnr, ssim = [], []

    for idx, el in enumerate(dataset):
        rgb_lr = np.asarray(el[lr].convert('RGB'), dtype=np.float32)
        bgr_lr = cv2.cvtColor(rgb_lr, cv2.COLOR_RGB2BGR)

        rgb_hr = np.asarray(el[hr].convert('RGB'), dtype=np.float32)
        bgr_hr = cv2.cvtColor(rgb_hr, cv2.COLOR_RGB2BGR)

        outscale = int(bgr_hr.shape[0] / bgr_lr.shape[0])
        res_hr = predict(
            bgr_lr,
            upsampler,
            outscale=outscale,
            use_face_enhancer=False
        )

        res_hr, bgr_hr = torch.from_numpy(res_hr), torch.from_numpy(bgr_hr)
        res_hr = res_hr.unsqueeze(0).permute(0, 3, 1, 2)
        bgr_hr = bgr_hr.unsqueeze(0).permute(0, 3, 1, 2)

        if torch.cuda.is_available():
            res_hr = res_hr.cuda()
            bgr_hr = bgr_hr.cuda()

        it_psnr = piq.psnr(res_hr, bgr_hr, data_range=255.0).item()
        it_ssim = piq.ssim(res_hr, bgr_hr, data_range=255.0).item()

        psnr += [it_psnr]
        ssim += [it_ssim]

        print(
            f'image = {idx+1}, '
            f'PSNR = {it_psnr:.3f}, '
            f'SSIM = {it_ssim:.3f}'
        )

    psnr_total = sum(psnr) / len(psnr)
    ssim_total = sum(ssim) / len(ssim)

    with open(save_path, 'a') as csv_file:
        csv_writer = csv.writer(csv_file)
        time = datetime.now()
        csv_writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S'),
            real_esrgan_config_name,
            project_type,
            project_name,
            lr[1:],
            hr[1:],
            split_config,
            f'{psnr_total:.3f}',
            f'{ssim_total:.3f}'
        ])


if __name__ == '__main__':
    root = '/home/epishchik/SR-Gaming-Bench'
    main(root)
