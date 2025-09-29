from image_utils.video_utils import write_video
from image_utils import Im
import numpy as np
import pytest
from pathlib import Path

img_path = Path("tests/high_res.png")
save_path = Path(__file__).parent / "output"


@pytest.mark.parametrize("fps", [1, 10])
@pytest.mark.parametrize("frames", [1, 10])
def test_write_video(frames, fps):
    img = Im.open(img_path).np
    video = np.stack([img for _ in range(frames)], axis=0)
    write_video(video, save_path / "test.mp4", fps=fps)
