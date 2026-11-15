import torch

from utils.video_utils import VideoWriter, get_video_info, read_video


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch version: {torch.__version__}")
    print(f"Using device: {device}")
    return device


def main():
    video_path = "input_videos/full.mp4"
    output_video_path = "output_videos/output.mp4"

    info = get_video_info(video_path)
    print(f"Input: {info.width}x{info.height} @ {info.fps:.2f} fps, {info.frame_count} frames")

    # Stream: read one frame -> process -> write -> next. Memory stays flat for any video length.
    n = 0
    with VideoWriter(output_video_path, info.fps, info.width, info.height) as writer:
        for frame in read_video(video_path):
            # TODO: detection / tracking / drawing on `frame` goes here
            writer.write(frame)
            n += 1

    print(f"Wrote {n} frames to {output_video_path}")


if __name__ == "__main__":
    device = get_device()
    main()
