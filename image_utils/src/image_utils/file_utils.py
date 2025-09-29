import pickle
import random
import shutil
import string
from datetime import datetime
from io import BytesIO
from pathlib import Path


def delete_create_folder(path: Path):
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def get_rand_hex() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=5))


def get_date_time_str() -> str:
    return datetime.now().strftime("%Y_%m_%d-%H_%M")


def strip_unsafe(filename):
    return "".join([c for c in filename if c.isalpha() or c.isdigit() or c == "" or c == "_"]).rstrip()


def save_pickle(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_pickle(obj, file_path: Path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def get_file_list(**kwargs):
    [f for f in get_files(**kwargs)]


def get_files(path: Path, recursive: bool = False, return_folders: bool = False, allowed_extensions=None):
    path = Path(path)

    if allowed_extensions or recursive:
        glob_str = "*" if allowed_extensions is None else f"*.[{''.join(allowed_extensions)}]*"
        iterator = path.rglob(glob_str) if recursive else path.glob(glob_str)
    else:
        iterator = path.iterdir()

    for file in iterator:
        if file.is_file() or return_folders:
            yield file


def get_images(path: Path, recursive: bool = False, allowed_extensions=[".png", ".jpg", ".jpeg"]):
    from image_utils import Im

    for file in get_files(path=path, recursive=recursive, allowed_extensions=allowed_extensions):
        yield Im.open(file)


def load_cached_from_url(url: str, cache: bool = True) -> BytesIO:
    import hashlib

    cache_dir = Path.home() / ".cache" / "image_utils"
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = hashlib.md5(url.encode()).hexdigest()
    local_path = cache_dir / filename

    if cache and local_path.exists():
        return BytesIO(local_path.read_bytes())
    else:
        image_bytesio = download_file_bytes(url)
        print(f"Downloading image from {url} and caching in {local_path}")
        with open(local_path, "wb") as file:
            file.write(image_bytesio.getvalue())
        return image_bytesio


def download_file_bytes(url) -> BytesIO:
    from urllib import error, request

    try:
        with request.urlopen(url) as response:
            return BytesIO(response.read())
    except error.URLError as e:
        raise Exception("Error downloading the image: " + str(e))
