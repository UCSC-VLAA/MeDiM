from typing import Union
from image_utils import Im, delete_create_folder, library_ops
from PIL import Image
import torch
import numpy as np
import pytest
from pathlib import Path
from einops import rearrange


def pytest_sessionstart(session):
    delete_create_folder(Path(__file__).parent / "output")
