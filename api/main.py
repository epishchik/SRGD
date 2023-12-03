import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

app = FastAPI()


@app.get('/info')
def info() -> FileResponse:
    info_file = FileResponse(
        path='./extra/info.txt',
        filename='info.txt',
        media_type='text'
    )
    return info_file


@app.post('/predict_example')
def predict_example(model: str, upscale: float) -> None:
    if upscale <= 1.0:
        raise HTTPException(
            status_code=480,
            detail='upscale parameter should be > 1.0'
        )

    img_np = cv2.imread('./extra/example.png')
    h, w = img_np.shape[0], img_np.shape[1]

    if int(h * upscale) > 4320 or int(w * upscale) > 7680:
        raise HTTPException(
            status_code=481,
            detail='potential output resolution is too large, max '
            'possible resolution is (4320, 7680) pixels in (h, w) format.'
        )

    print(img_np.shape, model, upscale)


@app.post('/predict_single')
def predict_single(image: UploadFile, model: str, upscale: float) -> None:
    if upscale <= 1.0:
        raise HTTPException(
            status_code=480,
            detail='upscale parameter should be > 1.0'
        )

    raw = np.fromstring(image.file.read(), np.uint8)
    img_np = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    h, w = img_np.shape[0], img_np.shape[1]

    if int(h * upscale) > 4320 or int(w * upscale) > 7680:
        raise HTTPException(
            status_code=481,
            detail='potential output resolution is too large, max '
            'possible resolution is (4320, 7680) pixels in (h, w) format.'
        )

    print(img_np.shape, model, upscale)


@app.post('/predict_multiple')
def predict_multiple(
    images: list[UploadFile],
    model: str,
    upscale: float
) -> list[None]:
    output = []
    for image in images:
        output += [predict_single(image, model, upscale)]
    return output


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)
