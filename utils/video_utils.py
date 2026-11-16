import cv2
import logging
# from exception import CustomException
import os
from .logger import log_step

def read_video(video_path):
    """
    Reads a video file and returns a list of frames.

    Args:
        video_path (str): Path to the video file.
"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def read_video_batch(video_path, batch_size=20, start_frame=0):
    """
    Reads only `batch_size` frames starting at `start_frame` (useful to test on a small sample).

    Args:
        video_path (str): Path to the video file.
        batch_size (int): Number of frames to read.
        start_frame (int): Index of the first frame to read.
    """
    with log_step("  Read frames", start_frame=start_frame, batch_size=batch_size) as out:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        while len(frames) < batch_size:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        out["frames_read"] = len(frames)
        out["frame_shape"] = frames[0].shape if frames else None
    return frames

def save_video(output_video_frame, output_video_path, fps=24, writer=None):
    """
    Saves a list of frames as a video file.

    To write in batches, pass the returned writer back in so frames are appended
    instead of overwriting the file, then call writer.release() once at the end.

    Args:
        output_video_frame (list): List of frames to save.
        output_video_path (str): Path to save the video file.
        fps (int): Frames per second for the output video.
        writer: Writer returned by a previous call (None = create a new video file).
    """
    with log_step("  Write video", frames=len(output_video_frame)) as out:
        if len(output_video_frame) == 0:
            raise ValueError("No frames to save.")

        if writer is None:
            height, width = output_video_frame[0].shape[:2]  # numpy frame shape is (height, width, channels)
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Define the codec for the output video
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise IOError(f"Could not open video writer for {output_video_path}")
            out["new_file"] = f"{output_video_path} | {width}x{height} @ {fps} fps"

        for frame in output_video_frame:
            writer.write(frame)
        out["frames_written"] = len(output_video_frame)
    return writer

def check_inputs(video_path, model_path, output_video_path, total_frames):
    """Fail fast: check every input path BEFORE loading the model. Returns (fps, total_frames)."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)          # does NOT raise on a bad file -> check isOpened()
    if not cap.isOpened():
        cap.release()
        raise IOError(f"OpenCV can't open the video (bad file or codec): {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        raise ValueError(f"Video reports fps={fps}: {video_path}")
    logging.info(f"Video OK: {video_path} | fps={fps:.2f} | frames={video_frames}")

    # FRAME_COUNT can be 0 for some files -> only clamp when it's known
    if 0 < video_frames < total_frames:
        logging.warning(f"total_frames={total_frames} but video has {video_frames} -> using {video_frames}")
        total_frames = video_frames

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    logging.info(f"Model OK: {model_path}")

    # cv2.VideoWriter fails silently if the folder is missing -> create it
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    return fps, total_frames