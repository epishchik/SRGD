import cv2
import os


def main(video, folder):
    os.makedirs(folder, exist_ok=True)

    cap = cv2.VideoCapture(video)
    cnt, success, total = 0, True, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while success:
        success, image = cap.read()
        if not success:
            break
        save_path = os.path.join(folder, f'{cnt}.png')
        cv2.imwrite(save_path, image)
        print(f'{cnt + 1} / {total}')
        cnt += 1


if __name__ == '__main__':
    video, folder = 'name.mp4', 'folder'
    main(video, folder)
