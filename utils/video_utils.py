"""Streaming video I/O: frames are read and written one at a time, never all held in memory."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def get_video_info(video_path: str) -> VideoInfo:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video {video_path}")
    info = VideoInfo(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def read_video(video_path: str) -> Iterator[np.ndarray]:
    """Yield frames one by one (BGR). Use in a for-loop; only one frame is in memory at a time."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


class VideoWriter:
    """Write frames one by one. Use as a context manager so the file is always finalized:

        with VideoWriter("out.mp4", info.fps, info.width, info.height) as writer:
            for frame in read_video("in.mp4"):
                writer.write(frame)
    """

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.size = (width, height)
        self._out = cv2.VideoWriter(output_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, self.size)
        if not self._out.isOpened():
            raise IOError(f"Could not open video writer for {output_path}")

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            raise ValueError(f"Frame size {w}x{h} does not match writer size {self.size[0]}x{self.size[1]}")
        self._out.write(frame)

    def close(self) -> None:
        self._out.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
