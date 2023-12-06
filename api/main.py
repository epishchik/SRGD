import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from real_esrgan import configure as configure_real_esrgan
from real_esrgan import predict as predict_real_esrgan

from utils.parse import parse_yaml

config_path = './configs/real_esrgan.yaml'
config = parse_yaml(config_path)

app = FastAPI()
upsampler = configure_real_esrgan(config)


@app.get('/info')
def info() -> FileResponse:
    info_file = FileResponse(
        path='./extra/info.txt',
        filename='info.txt',
        media_type='text'
    )
    return info_file


@app.post('/upscale_example')
def upscale_example() -> Response:
    img = cv2.imread('./extra/example.png', cv2.IMREAD_UNCHANGED)

    h, w = img.shape[0], img.shape[1]
    outscale = config['outscale']

    h_up, w_up = int(h * outscale), int(w * outscale)
    if h_up > 4320 or w_up > 7680:
        raise HTTPException(
            status_code=481,
            detail=f'potential output resolution ({h_up, w_up}) is too large, '
            'max possible resolution is (4320, 7680) pixels in (h, w) format.'
        )

    out_img = predict_real_esrgan(img, upsampler, config)
    _, enc_img = cv2.imencode('.png', out_img)
    bytes_img = enc_img.tobytes()

    return Response(bytes_img, media_type='image/png')


@app.post('/upscale')
def upscale(image_file: UploadFile) -> Response:
    raw = np.fromstring(image_file.file.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    h, w = img.shape[0], img.shape[1]
    outscale = config['outscale']

    h_up, w_up = int(h * outscale), int(w * outscale)
    if h_up > 4320 or w_up > 7680:
        raise HTTPException(
            status_code=481,
            detail=f'potential output resolution ({h_up, w_up}) is too large, '
            'max possible resolution is (4320, 7680) pixels in (h, w) format.'
        )

    out_img = predict_real_esrgan(img, upsampler, config)
    _, enc_img = cv2.imencode('.png', out_img)
    bytes_img = enc_img.tobytes()

    return Response(bytes_img, media_type='image/png')


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)
