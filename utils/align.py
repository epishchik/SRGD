import os


def clean_from_single_folder(root: str, target_folder: str) -> None:
    folders = [
        os.path.join(root, resolution) for resolution in os.listdir(root)
    ]

    assert target_folder in folders
    target_images, extra_images = os.listdir(target_folder), []

    for raw_folder in folders:
        raw_images, delete_images = os.listdir(raw_folder), []

        for image in raw_images:
            if image not in target_images:
                delete_images += [os.path.join(raw_folder, image)]
        extra_images += delete_images

    for extra_image in extra_images:
        os.remove(extra_image)


def clean_all_folders(root: str) -> None:
    folders = os.listdir(root)
    target_folders = [os.path.join(root, folder) for folder in folders]

    for target_folder in target_folders:
        clean_from_single_folder(root, target_folder)


if __name__ == '__main__':
    root = ''
    clean_all_folders(root)
