import pandas as pd


def main() -> None:
    metrics = pd.read_csv('../dvc_data/metrics.csv')

    show_columns = [
        'time',
        'sr_model',
        'project_type',
        'project_name',
        'lr',
        'hr',
        'split',
        'psnr',
        'ssim',
        'lpips'
    ]

    query = ('(project_name == "ActionRPG" and split == "train") or '
             '(project_name == "CitySample" and split == "val")')

    if query:
        filtered_metrics = metrics.query(query)[show_columns]
    else:
        filtered_metrics = metrics[show_columns]

    print(filtered_metrics)


if __name__ == '__main__':
    main()
