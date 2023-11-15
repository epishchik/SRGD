import cv2
import os


def main(video: str, folder: str, prefix: str, save_every: int = 1) -> None:
    os.makedirs(folder, exist_ok=True)

    cap = cv2.VideoCapture(video)
    cnt, success, total = 0, True, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while success:
        success, image = cap.read()
        if not success:
            break
        cnt += 1
        if cnt % save_every == 0:
            save_path = os.path.join(folder, f'{prefix}_{cnt}.png')
            cv2.imwrite(save_path, image)
        print(f'{cnt} / {total}')


if __name__ == '__main__':
    video = ''
    folder = ''
    prefix, save_every = '', 5

    main(video, folder, prefix, save_every)
