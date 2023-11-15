import random
import numpy as np
import math
import os
import cv2

from basicsr.data.degradations import circular_lowpass_kernel, \
    random_mixed_kernels


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def degradation(h: int):
    kernel_size1 = math.ceil(270 / h * 21)
    kernel_size2 = math.ceil(270 / h * 21)
    final_kernel_size = math.ceil(270 / h * 21)

    kernel_list1 = [
        'iso',
        'aniso',
        'generalized_iso',
        'generalized_aniso',
        'plateau_iso',
        'plateau_aniso'
    ]

    kernel_list2 = [
        'iso',
        'aniso',
        'generalized_iso',
        'generalized_aniso',
        'plateau_iso',
        'plateau_aniso'
    ]

    kernel_prob1 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]

    blur_sigma1 = [0.2, 3]
    blur_sigma2 = [0.2, 1.5]

    betag_range1 = [0.5, 4]
    betag_range2 = [0.5, 4]

    betap_range1 = [1, 2]
    betap_range2 = [1, 2]

    if np.random.uniform() < 0.1:
        if kernel_size1 < 13:
            omega_c1 = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c1 = np.random.uniform(np.pi / 5, np.pi)
        kernel1 = circular_lowpass_kernel(omega_c1, kernel_size1, pad_to=False)
    else:
        kernel1 = random_mixed_kernels(
            kernel_list1,
            kernel_prob1,
            kernel_size1,
            blur_sigma1,
            blur_sigma1,
            [-math.pi, math.pi],
            betag_range1,
            betap_range1,
            noise_range=None
        )

    pad_size1 = (21 - kernel_size1) // 2
    kernel1 = np.pad(kernel1, ((pad_size1, pad_size1), (pad_size1, pad_size1)))

    if np.random.uniform() < 0.1:
        if kernel_size2 < 13:
            omega_c2 = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c2 = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c2, kernel_size2, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
            kernel_list2,
            kernel_prob2,
            kernel_size2,
            blur_sigma2,
            blur_sigma2,
            [-math.pi, math.pi],
            betag_range2,
            betap_range2,
            noise_range=None
        )

    pad_size2 = (21 - kernel_size2) // 2
    kernel2 = np.pad(kernel2, ((pad_size2, pad_size2), (pad_size2, pad_size2)))

    if np.random.uniform() < 0.8:
        final_omega_c = np.random.uniform(np.pi / 3, np.pi)
        final_sinc_kernel = circular_lowpass_kernel(
            final_omega_c,
            final_kernel_size,
            pad_to=21
        )
    else:
        final_sinc_kernel = np.zeros(21, 21).astype(np.float32)
        final_sinc_kernel[10, 10] = 1

    return kernel1, kernel2, final_sinc_kernel


def main(src_folder: str, dst_folder: str, h: int, w: int) -> None:
    src_files = os.listdir(src_folder)
    os.makedirs(dst_folder, exist_ok=True)

    for src_file in src_files:
        img = cv2.imread(src_file)
        kernel1, kernel2, final_sinc_kernel = degradation(h)


if __name__ == '__main__':
    src_folder = ''
    dst_folder = ''
    h, w = 540, 960
    seed = 42

    set_seed(seed)
    main(src_folder, dst_folder, h, w)
