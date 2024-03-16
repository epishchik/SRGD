import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from metric.metric_class import MetricSR


class MetricTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        map_metric_names = {
            "psnr": "psnr",
            "ssim": "ssim",
            "lpips": "lpips",
        }
        map_no_ref_metric_names = {}

        for metric_name, map_metric_name in map_no_ref_metric_names.items():
            map_metric_names[metric_name] = map_metric_name
        metric_names = [k for k in map_metric_names.keys()]
        no_ref_metric_names = [k for k in map_no_ref_metric_names.keys()]

        cls.metric_calculator_cpu = MetricSR(
            metric_names,
            no_ref_metric_names,
            map_metric_names,
            device="cpu",
        )

        cls.metric_calculator_cuda = MetricSR(
            metric_names,
            no_ref_metric_names,
            map_metric_names,
            device="cuda",
        )

        cls.metric_names = metric_names

    def test_single_image_perfect(self):
        ups_hr = real_hr = torch.rand((1, 3, 32, 32), dtype=torch.float32)

        self.metric_calculator_cuda.calculate(ups_hr, real_hr)
        self.metric_calculator_cpu.calculate(ups_hr, real_hr)

        best_psnr, best_ssim, best_lpips = 80.0, 1.0, 0.0

        assert self.metric_calculator_cuda.calculate_total("psnr") - best_psnr <= 1e-6
        assert self.metric_calculator_cpu.calculate_total("psnr") - best_psnr <= 1e-6

        assert self.metric_calculator_cuda.calculate_total("ssim") - best_ssim <= 1e-6
        assert self.metric_calculator_cpu.calculate_total("ssim") - best_ssim <= 1e-6

        assert self.metric_calculator_cuda.calculate_total("lpips") - best_lpips <= 1e-6
        assert self.metric_calculator_cpu.calculate_total("lpips") - best_lpips <= 1e-6

    def test_multiple_image_perfect(self):
        ups_hr = real_hr = torch.rand((8, 3, 32, 32), dtype=torch.float32)

        self.metric_calculator_cuda.calculate(ups_hr, real_hr)
        self.metric_calculator_cpu.calculate(ups_hr, real_hr)

        best_psnr, best_ssim, best_lpips = 80.0, 1.0, 0.0

        assert self.metric_calculator_cuda.calculate_total("psnr") - best_psnr <= 1e-6
        assert self.metric_calculator_cpu.calculate_total("psnr") - best_psnr <= 1e-6

        assert self.metric_calculator_cuda.calculate_total("ssim") - best_ssim <= 1e-6
        assert self.metric_calculator_cpu.calculate_total("ssim") - best_ssim <= 1e-6

        assert self.metric_calculator_cuda.calculate_total("lpips") - best_lpips <= 1e-6
        assert self.metric_calculator_cpu.calculate_total("lpips") - best_lpips <= 1e-6
