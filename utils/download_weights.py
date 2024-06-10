import io
import os.path
import zipfile
from argparse import ArgumentParser
from pathlib import Path
from urllib.parse import urlencode

import requests


def download_file(pk: str, local_name: str, src_type: str) -> None:
    """
    Download file / folder from Yandex Disk using API.

    Parameters
    ----------
    pk : str
        Public link to the folder or file from Yandex Disk.
    local_name : str
        Path to save file (with filename) or folder.
    src_type : str
        Source type [folder, file].

    Returns
    -------
    None
    """
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"

    final_url = base_url + urlencode(dict(public_key=pk))
    response = requests.get(final_url)
    download_url = response.json()["href"]

    download_response = requests.get(download_url)
    if src_type == "file":
        with open(local_name, "wb") as f:
            f.write(download_response.content)
            print(f"File {local_name} downloaded")
    elif src_type == "folder":
        zip_file = zipfile.ZipFile(io.BytesIO(download_response.content))
        zip_file.extractall(local_name)
        print("Folder downloaded")


def download(download_source: str, save_folder: str) -> None:
    """
    Download data from Yandex Disk using API.

    Parameters
    ----------
    download_source : str
        URL to download from.
    save_folder : str
        Path to save folder.

    Returns
    -------
    None
    """
    if save_folder:
        save_folder = Path(save_folder)
        os.makedirs(save_folder, exist_ok=True)
    else:
        save_folder = Path.cwd().parent

    files = {
        save_folder: (download_source, "folder"),
    }

    print(f"Will be downloaded {len(files)} sources")
    for name, (src_link, src_type) in files.items():
        download_file(src_link, str(name), src_type)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-ds",
        "--download-source",
        type=str,
        required=True,
        help="URL to download from",
    )
    parser.add_argument(
        "-sf",
        "--save-folder",
        type=str,
        default=None,
        help="path to save folder",
    )
    args = parser.parse_args()

    download(args.download_source, args.save_folder)
