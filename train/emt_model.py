import datetime
import logging
import os
import time
from pathlib import Path

import emt.archs  # noqa
import emt.data  # noqa
import emt.models  # noqa
import torch
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.train import (
    create_train_val_dataloader,
    init_tb_loggers,
    load_resume_state,
)
from basicsr.utils import (
    AvgTimer,
    MessageLogger,
    get_env_info,
    get_root_logger,
    get_time_str,
    make_exp_dirs,
    mkdir_and_rename,
)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options

import mlflow


def train() -> None:
    """
    Обучение модели EMT. Все параметры прописываются в конфиге .yaml.
    Путь к конфигу передается через параметр -opt.

    Returns
    -------
    None
    """
    root = Path(__file__).parents[1]
    exp_folder = root / "experiments"
    os.makedirs(exp_folder, exist_ok=True)

    opt, args = parse_options(root, is_train=True, options_in_root=True)
    opt["root_path"] = root

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    resume_state = load_resume_state(opt)

    if resume_state is None:
        make_exp_dirs(opt)
        if (
            opt["logger"].get("use_tb_logger")
            and "debug" not in opt["name"]
            and opt["rank"] == 0
        ):
            mkdir_and_rename(os.path.join(exp_folder, "tb_logger", opt["name"]))

    copy_opt_file(root / args.opt, opt["path"]["experiments_root"])

    log_file = os.path.join(
        opt["path"]["log"], f"train_{opt['name']}_{get_time_str()}.log"
    )

    logger = get_root_logger(
        logger_name="basicsr", log_level=logging.INFO, log_file=log_file
    )
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    tb_logger = init_tb_loggers(opt)

    # TODO: создать даталоадер для datasets.type == "huggingface"
    datasets_type = opt["datasets"]["type"]
    if datasets_type == "files":
        result = create_train_val_dataloader(opt, logger, datasets_type, root=root)
        train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    model = build_model(opt, root=root)
    if resume_state:
        model.resume_training(resume_state)
        logger.info(
            f"Resuming training from epoch: {resume_state['epoch']},"
            f"iter: {resume_state['iter']}."
        )
        start_epoch = resume_state["epoch"]
        current_iter = resume_state["iter"]
    else:
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    prefetch_mode = opt["datasets"][datasets_type]["train"].get("prefetch_mode")
    if prefetch_mode is None or prefetch_mode == "cpu":
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == "cuda":
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f"Use {prefetch_mode} prefetch dataloader")
        if opt["datasets"][datasets_type]["train"].get("pin_memory") is not True:
            raise ValueError("Please set pin_memory=True for CUDAPrefetcher.")
    else:
        raise ValueError(
            f"Wrong prefetch_mode {prefetch_mode}. "
            f"Supported ones are: None, 'cuda', 'cpu'."
        )

    mlflow_opt = opt["logger"]["mlflow"]
    if mlflow_opt["tracking_uri"]:
        mlflow.set_tracking_uri(mlflow_opt["tracking_uri"])
        if not mlflow.get_experiment_by_name(mlflow_opt["experiment"]):
            mlflow.create_experiment(mlflow_opt["experiment"])
        mlflow.set_experiment(mlflow_opt["experiment"])

        mlflow.start_run(
            run_name=mlflow_opt["run"],
            log_system_metrics=mlflow_opt["log_system_metrics"],
        )

        mlflow.log_params(opt)

    logger.info(f"Start training from epoch: {start_epoch}, iter: {current_iter}")
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break

            model.update_learning_rate(
                current_iter, warmup_iter=opt["train"].get("warmup_iter", -1)
            )

            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                msg_logger.reset_start_time()
            if current_iter % opt["logger"]["print_freq"] == 0:
                log_vars = {"epoch": epoch, "iter": current_iter}
                log_vars.update({"lrs": model.get_current_learning_rate()})
                log_vars.update(
                    {
                        "time": iter_timer.get_avg_time(),
                        "data_time": data_timer.get_avg_time(),
                    }
                )
                train_losses = model.get_current_log()
                log_vars.update(train_losses)
                msg_logger(log_vars)
                if mlflow_opt["tracking_uri"]:
                    for k, v in train_losses.items():
                        mlflow.log_metric(k, v, step=current_iter)

            if current_iter % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Saving models and training states.")
                model.save(epoch, current_iter)
                if mlflow_opt["tracking_uri"]:
                    net_g_path = os.path.join(
                        opt["path"]["models"],
                        opt["network_prefix"]["network_g"] + f"_{current_iter}.pth",
                    )

                    net_d_path = os.path.join(
                        opt["path"]["models"],
                        opt["network_prefix"]["network_d"] + f"_{current_iter}.pth",
                    )

                    state_path = os.path.join(
                        opt["path"]["training_states"], f"{current_iter}.state"
                    )

                    mlflow.log_artifact(net_g_path)
                    mlflow.log_artifact(net_d_path)
                    mlflow.log_artifact(state_path)

            if opt.get("val") is not None and (
                current_iter % opt["val"]["val_freq"] == 0
            ):
                if len(val_loaders) > 1:
                    logger.warning(
                        "Multiple validation datasets are *only* supported by SRModel."
                    )
                for val_loader_ind, val_loader in enumerate(val_loaders):
                    model.validation(
                        val_loader,
                        current_iter,
                        tb_logger,
                        opt["val"]["save_img"],
                    )
                    if mlflow_opt["tracking_uri"]:
                        for metric, value in model.metric_results.items():
                            mlflow.log_metric(
                                f"val{val_loader_ind}_{metric}",
                                value,
                                step=current_iter,
                            )

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"End of training. Time consumed: {consumed_time}")
    logger.info("Save the latest model.")
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get("val") is not None:
        for val_loader in val_loaders:
            model.validation(
                val_loader, current_iter, tb_logger, opt["val"]["save_img"]
            )
    if tb_logger:
        tb_logger.close()


if __name__ == "__main__":
    train()
