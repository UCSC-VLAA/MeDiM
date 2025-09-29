from image_utils import get_files
import pytest
from pathlib import Path

folder_path = Path("tests")


@pytest.mark.parametrize("recursive", [True, False])
@pytest.mark.parametrize("return_folders", [True, False])
@pytest.mark.parametrize("allowed_extensions", [None, ["png"], ["png", "jpg"]])
def test_get_files(recursive, return_folders, allowed_extensions):
    for file in get_files(folder_path, recursive, return_folders, allowed_extensions):
        print(file)
