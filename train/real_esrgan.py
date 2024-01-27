import logging
import os
from pathlib import Path

import torch
from basicsr.models import build_model
from basicsr.train import (
    MessageLogger,
    init_tb_loggers,
    load_resume_state,
    make_exp_dirs,
    mkdir_and_rename,
)
from basicsr.utils import get_env_info, get_root_logger, get_time_str
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def train_from_hf() -> None:
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
    # if opt["datasets"]["type"] == "files":
    #     result = create_train_val_dataloader(opt, logger)
    #     train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    model = build_model(opt)
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

    print(msg_logger, start_epoch)


if __name__ == "__main__":
    train_from_hf()
