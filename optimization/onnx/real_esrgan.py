import os
import sys
from argparse import ArgumentParser
from pathlib import Path, PurePath

import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from model.real_esrgan import configure
from utils.parse import parse_yaml


def torch2onnx(root: PurePath, save_path: str, model_config_path: str) -> None:
    """
    Конвертация torch Real-ESRGAN модели в onnx формат.

    Parameters
    ----------
    root : PurePath
        Путь к корню проекта.
    save_path : str
        Относительный путь к создаваемому .onnx файлу модели.
    model_config_path : str
        Относительный путь к конфигурационному файлу модели.

    Returns
    -------
    None
    """
    save_folder = root / Path(save_path).parent
    os.makedirs(save_folder, exist_ok=True)

    model_config = root / model_config_path
    model_config_dct = parse_yaml(str(model_config))

    upsampler = configure(root, model_config_dct)
    torch_model, device, half = upsampler.model, upsampler.device, upsampler.half
    dtype = torch.float16 if half else torch.float32

    torch_model.eval()

    h_in, w_in = 64, 64
    dummy_input = torch.rand((1, 3, h_in, w_in), dtype=dtype).to(device)
    dummy_output = torch_model(dummy_input)
    h_out = dummy_output.shape[2]
    print(f"h_in={h_in}, h_out={h_out}")
    upscale = h_out / h_in

    torch.onnx.export(
        torch_model,
        dummy_input,
        str(root / save_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["lr"],
        output_names=["hr"],
        dynamic_axes={
            "lr": {2: "h", 3: "w"},
            "hr": {2: f"{upscale:.1f}*h", 3: f"{upscale:.1f}*w"},
        },
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    # TODO добавить help
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    args = parser.parse_args()

    root = Path(__file__).parents[2]
    torch2onnx(root, args.save_path, args.model_config)
