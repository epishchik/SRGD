import os
import shutil
from typing import Any

import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def configure(root: str, config: dict[str, Any]) -> Any:
    model, model_path, netscale, dni_weight = None, None, None, None

    model_path = os.path.join(root, config['weights'])
    model_name = config['filename']
    outscale = config['outscale']
    denoise_strength = config['denoise_strength']
    use_face_enhancer = config['use_face_enhancer']
    tile = config['tile']
    tile_pad = config['tile_pad']
    pre_pad = config['pre_pad']
    fp32 = config['fp32']
    gpu_id = config['gpu_id']

    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        netscale = 4
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        netscale = 4
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=4
        )
        netscale = 4
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2
        )
        netscale = 2
    elif model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=16,
            upscale=4,
            act_type='prelu'
        )
        netscale = 4
    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type='prelu'
        )
        netscale = 4

    if model_name == 'realesr-general-x4v3' and denoise_strength < 1.0:
        wdn_model_path = os.path.join(root, config['wdn_weights'])
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1.0 - denoise_strength]

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id
    )

    if use_face_enhancer:
        shutil.copytree(
            os.path.join(root, config['GFPGAN_weights']['additional']),
            os.path.join(os.getcwd(), 'gfpgan'),
            dirs_exist_ok=True
        )

        face_enhancer = GFPGANer(
            model_path=os.path.join(config['GFPGAN_weights']['model']),
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )
        return face_enhancer
    return upsampler


def predict(
    img: np.ndarray,
    upsampler: Any,
    outscale: float = 4.0,
    use_face_enhancer: bool = False
) -> np.ndarray:
    if use_face_enhancer:
        _, _, out_img = upsampler.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
    else:
        out_img, _ = upsampler.enhance(img, outscale=outscale)

    return out_img
