import uvicorn
from fastapi import FastAPI
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


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)
