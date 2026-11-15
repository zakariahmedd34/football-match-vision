import cv2

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
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    while len(frames) < batch_size:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
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
    if len(output_video_frame) == 0:
        raise ValueError("No frames to save.")

    if writer is None:
        height, width = output_video_frame[0].shape[:2]  # numpy frame shape is (height, width, channels)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Define the codec for the output video
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise IOError(f"Could not open video writer for {output_video_path}")

    for frame in output_video_frame:
        writer.write(frame)
    return writer
