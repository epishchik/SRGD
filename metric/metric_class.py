import piq
import torch.cuda


class MetricSR:
    """
    Class for calculating SR metrics.

    Attributes
    ----------
    metric_names : list[str]
        List of all metric names.
    no_ref_metric_names : list[str]
        List of No-Reference (NR) metric names.
    map_metric_names : dict[str, str]
        Mapping of metric names to function names.
    metric_history : dict[str, list[float]]
        History values for all metrics.

    Methods
    -------
    calculate(upscaled_hr, real_hr)
        Calculate all metrics between upscaled HR and real HR.
    calculate_total(metric_name)
        Calculate mean value of specific metric accross all history.
    """

    def __init__(
        self,
        metric_names: list[str],
        no_ref_metric_names: list[str],
        map_metric_names: dict[str, str],
        device: str = None,
    ) -> None:
        """
        Constructor of MetricSR class.

        Parameters
        ----------
        metric_names : list[str]
            List of all metric names.
        no_ref_metric_names : list[str]
            List of No-Reference (NR) metric names.
        map_metric_names : dict[str, str]
            Mapping of metric names to function names.
        device : str
            Device to use for calculations.

        Returns
        -------
        None
        """
        self.metric_names = metric_names
        self.map_metric_names = map_metric_names

        self_metric_names = []
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if "lpips" in self.metric_names:
            self.lpips = piq.LPIPS().to(self.device)
            self_metric_names += ["lpips"]

        if "dists" in self.metric_names:
            self.dists = piq.DISTS().to(self.device)
            self_metric_names += ["dists"]

        self.self_metric_names = self_metric_names
        self.no_ref_metric_names = no_ref_metric_names
        self.metric_history = {name: [] for name in self.metric_names}

    def calculate(
        self, upscaled_hr: torch.FloatTensor, real_hr: torch.FloatTensor
    ) -> None:
        """
        Calculate all metrics between upscaled HR and real HR.

        Parameters
        ----------
        upscaled_hr : torch.FloatTensor
            Upscaled HR image in (1, c, h, w) format.
        real_hr : torch.FloatTensor
            Reference HR image in (1, c, h, w) format.

        Returns
        -------
        None
        """
        upscaled_hr = upscaled_hr.to(self.device)
        real_hr = real_hr.to(self.device)

        for metric_name in self.metric_names:
            if metric_name not in self.no_ref_metric_names:
                if metric_name in self.self_metric_names:
                    metric_val = getattr(self, metric_name)(upscaled_hr, real_hr).item()
                else:
                    metric_func = getattr(piq, self.map_metric_names[metric_name])
                    metric_val = metric_func(
                        upscaled_hr, real_hr, data_range=1.0
                    ).item()
            else:
                if metric_name in self.self_metric_names:
                    metric_val = getattr(self, metric_name)(upscaled_hr).item()
                else:
                    metric_func = getattr(piq, self.map_metric_names[metric_name])
                    if metric_name != "tv":
                        metric_val = metric_func(upscaled_hr, data_range=1.0).item()
                    else:
                        metric_val = metric_func(upscaled_hr).item()
            self.metric_history[metric_name] += [metric_val]

    def calculate_total(self, metric_name: str) -> float:
        """
        Calculate mean value of specific metric accross all history.

        Parameters
        ----------
        metric_name : str
            Metric name.

        Returns
        -------
        float
            Calculated metric value.
        """
        metric_sum = sum(self.metric_history[metric_name])
        metric_len = len(self.metric_history[metric_name])
        metric = metric_sum / metric_len
        return metric
