import os


def rename(root: str) -> None:
    '''
        Переименование файлов со всех папок внутри root,
        файлы с одинаковым названием получат новое одинаковое название.

        Parameters
        ----------
        root : str
            Путь к корню, содержащий все папки,
            файлы внутри которых нужно переименовать.

        Returns
        -------
        None
    '''
    folders = [os.path.join(root, folder) for folder in os.listdir(root)]

    files = os.listdir(folders[0])
    map_names = {
        k: str(v+1).zfill(5) for v, k in enumerate(files)
    }

    for i, folder in enumerate(folders):
        for src_name, dst_name in map_names.items():
            ext = src_name.split('.')[-1]
            src_name = os.path.join(folder, src_name)
            dst_name = os.path.join(folder, f'{dst_name}.{ext}')
            os.rename(src_name, dst_name)


if __name__ == '__main__':
    root = ''
    rename(root)
