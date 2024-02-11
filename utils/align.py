import os
from argparse import ArgumentParser


def clean_from_single_folder(root: str, target_folder: str) -> None:
    """
    Очистка всех папок из root от файлов без пары в папке target_folder.

    Parameters
    ----------
    root : str
        Путь к корню, все папки из root будут сравниваться
        с target_folder по файлам.
    target_folder : str
        Путь к папке, все файлы из папок из root будут удалены,
        если их нет в target_folder.

    Returns
    -------
    None
    """
    folders = [os.path.join(root, resolution) for resolution in os.listdir(root)]

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
    """
    Функция попарной очистки всех файлов из всех
    папок из root от файлов без пары.

    Parameters
    ----------
    root : str
        Путь к корню, все папки из root будут поданы в попарную очистку.

    Returns
    -------
    None
    """
    folders = os.listdir(root)
    target_folders = [os.path.join(root, folder) for folder in folders]

    for target_folder in target_folders:
        clean_from_single_folder(root, target_folder)


if __name__ == "__main__":
    parser = ArgumentParser()
    # TODO добавить help
    parser.add_argument("-r", "--root", type=str, required=True)
    args = parser.parse_args()

    clean_all_folders(args.root)
