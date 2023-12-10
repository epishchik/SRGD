import piq
import torch.cuda


class MetricSR:
    """
        Класс для вычисления метрик.

        Attributes
        ----------
        metric_names : list[str]
            Список названий всех метрик.
        no_ref_metric_names : list[str]
            Список названий метрик, работающих без reference изображения.
        map_metric_names : dict[str, str]
            Словарь маппинга из собственного названия метрики в название
            функции реализующей эту метрику.
        metric_history : dict[str, list[float]]
            История значений каждой метрики.

        Methods
        -------
        calculate(upscaled_hr, real_hr)
            Вычисление всех метрик между сгенерированным HR и reference HR.
        calculate_total(metric_name)
            Вычисление среднего значения определенной метрики по всей истории.
    """

    def __init__(
        self,
        metric_names: list[str],
        no_ref_metric_names: list[str],
        map_metric_names: dict[str, str]
    ) -> None:
        """
            Конфигурация класса для вычисления метрик.

            Parameters
            ----------
            metric_names : list[str]
                Список названий всех метрик.
            no_ref_metric_names : list[str]
                Список названий метрик, работающих без reference изображения.
            map_metric_names : dict[str, str]
                Словарь маппинга из собственного названия метрики в название
                функции реализующей эту метрику.

            Returns
            -------
            None
        """
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

    def calculate(
        self,
        upscaled_hr: torch.FloatTensor,
        real_hr: torch.FloatTensor
    ) -> None:
        """
            Вычисление всех метрик между сгенерированным HR и reference HR.

            Parameters
            ----------
            upscaled_hr : torch.FloatTensor
                Сгенерированное HR изображение в формате (1, c, h, w).
            real_hr : torch.FloatTensor
                Reference HR изображение в формате (1, c, h, w).

            Returns
            -------
            None
        """
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

    def calculate_total(self, metric_name: str) -> float:
        """
            Вычисление среднего значения определенной метрики по всей истории.

            Parameters
            ----------
            metric_name : str
                Название метрики.

            Returns
            -------
            float
                Значение указанной метрики.
        """
        metric_sum = sum(self.metric_history[metric_name])
        metric_len = len(self.metric_history[metric_name])
        metric = metric_sum / metric_len
        return metric
