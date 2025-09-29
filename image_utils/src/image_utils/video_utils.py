import numpy as np
from pathlib import Path
from typing import Final, Set
from enum import Enum, unique

# Several functions were taken from: https://github.com/argoverse/av2-api


@unique
class VideoCodecs(str, Enum):
    """Available video codecs for encoding mp4 videos.

    NOTE: The codecs available are dependent on the FFmpeg build that
        you are using. We recommend defaulting to LIBX264.
    """

    LIBX264 = "libx264"  # https://en.wikipedia.org/wiki/Advanced_Video_Coding
    LIBX265 = "libx265"  # https://en.wikipedia.org/wiki/High_Efficiency_Video_Coding
    HEVC_VIDEOTOOLBOX = "hevc_videotoolbox"  # macOS GPU acceleration.


HIGH_EFFICIENCY_VIDEO_CODECS: Final[Set[VideoCodecs]] = set([VideoCodecs.LIBX265, VideoCodecs.HEVC_VIDEOTOOLBOX])


def crop_video_to_even_dims(video: np.ndarray) -> np.ndarray:
    """Crop a video tensor (4d nd-array) along the height and width dimensions to assure even dimensions.

    Note: typical "pad" or "crop" filters do not function properly with pypi AV's stream configuration options.

    Args:
        video: (N,H1,W1,3) array representing N RGB frames of identical dimensions, where H1 and W1 may be odd.

    Returns:
        (N,H2,W2,3) array representing N RGB frames of identical dimensions, where H2 and W2 are even.
            The crop is performed on the far right column and/or bottom row of each frame.
    """
    _, H1, W1, _ = video.shape
    height_crop_sz = H1 % 2
    width_crop_sz = W1 % 2

    H2 = H1 - height_crop_sz
    W2 = W1 - width_crop_sz

    return video[:, :H2, :W2, :]


def write_video(
    video: np.ndarray,
    dst: Path,
    codec: VideoCodecs = VideoCodecs.LIBX264,
    fps: int = 10,
    crf: int = 27,
    preset: str = "veryfast",
) -> None:
    """Use the FFMPEG Python bindings to encode a video from a sequence of RGB frames.

    Reference: https://github.com/PyAV-Org/PyAV

    Args:
        video: (N,H,W,3) Array representing N RGB frames of identical dimensions.
        dst: Path to save folder.
        codec: Name of the codec.
        fps: Frame rate for video.
        crf: Constant rate factor (CRF) parameter of video, controlling the quality.
            Lower values would result in better quality, at the expense of higher file sizes.
            For x264, the valid Constant Rate Factor (crf) range is 0-51.
        preset: File encoding speed. Options range from "ultrafast", ..., "fast", ..., "medium", ..., "slow", ...
            Higher compression efficiency often translates to slower video encoding speed, at file write time.
    """
    _, H, W, _ = video.shape

    # crop, if the height or width is odd (avoid "height not divisible by 2" error)
    if H % 2 != 0 or W % 2 != 0:
        video = crop_video_to_even_dims(video)
        _, H, W, _ = video.shape

    dst.parent.mkdir(parents=True, exist_ok=True)
    import av

    with av.open(str(dst), "w") as output:
        stream = output.add_stream(codec, fps)
        if codec in HIGH_EFFICIENCY_VIDEO_CODECS:
            stream.codec_tag = "hvc1"
        stream.width = W
        stream.height = H
        stream.options = {
            "crf": str(crf),
            "hwaccel": "auto",
            "movflags": "+faststart",
            "preset": preset,
            "profile:v": "main",
        }

        format = "rgb24"
        for _, img in enumerate(video):
            frame = av.VideoFrame.from_ndarray(img, format=format)
            output.mux(stream.encode(frame))
        output.mux(stream.encode(None))
