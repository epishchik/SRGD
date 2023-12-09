import piq
import torch.cuda


class MetricSR:
    def __init__(
        self,
        metric_names: list[str],
        no_ref_metric_names: list[str],
        map_metric_names: dict[str, str]
    ) -> None:
        self.metric_names = metric_names
        self.map_metric_names = map_metric_names

        self_metric_names = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if 'lpips' in self.metric_names:
            self.lpips = piq.LPIPS().to(device)
            self_metric_names += ['lpips']

        if 'dists' in self.metric_names:
            self.dists = piq.DISTS().to(device)
            self_metric_names += ['dists']

        self.self_metric_names = self_metric_names
        self.no_ref_metric_names = no_ref_metric_names
        self.metric_history = {name: [] for name in self.metric_names}

    def calculate(self, upscaled_hr, real_hr) -> None:
        for metric_name in self.metric_names:
            if metric_name not in self.no_ref_metric_names:
                if metric_name in self.self_metric_names:
                    metric_val = getattr(self, metric_name)(
                        upscaled_hr,
                        real_hr
                    ).item()
                else:
                    metric_func = getattr(
                        piq,
                        self.map_metric_names[metric_name]
                    )
                    metric_val = metric_func(
                        upscaled_hr,
                        real_hr,
                        data_range=1.0
                    ).item()
            else:
                if metric_name in self.self_metric_names:
                    metric_val = getattr(self, metric_name)(upscaled_hr).item()
                else:
                    metric_func = getattr(
                        piq,
                        self.map_metric_names[metric_name]
                    )
                    if metric_name != 'tv':
                        metric_val = metric_func(
                            upscaled_hr,
                            data_range=1.0
                        ).item()
                    else:
                        metric_val = metric_func(upscaled_hr).item()
            self.metric_history[metric_name] += [metric_val]

    def calculate_epoch(self, metric_name: str) -> float:
        metric_sum = sum(self.metric_history[metric_name])
        metric_len = len(self.metric_history[metric_name])
        metric = metric_sum / metric_len
        return metric
