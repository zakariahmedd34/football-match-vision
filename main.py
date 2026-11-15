import os
import pickle
import torch
from tqdm import tqdm
from utils.video_utils import read_video, read_video_batch, save_video
from trackers import Tracker
import json

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch version: {torch.__version__}")
    print(f"Using device: {device}")
    return device


def main():
    model_path = "models/best.pt"
    video_path = "input_videos/full.mp4"
    output_video_path = "output_videos/output.mp4"
    stub_path = "stubs/tracks.pkl"

    total_frames = 750
    batch_size = 100
    writer = None  # created on the first batch, then reused so each batch is appended

    # Tracks for the WHOLE video: all_tracks["players"][frame_num] -> {track_id: {"bbox": [...]}}
    # If a stub exists from a previous run, load it once and skip detection.
    tracker = Tracker(model_path)  # ONE tracker for all batches -> IDs continue across batches
    from_stub = os.path.exists(stub_path)

    if from_stub:
        with open(stub_path, "rb") as f:
            all_tracks = pickle.load(f)
        print(f"Loaded tracks from {stub_path}") 
    else:
        all_tracks = {"players": [], "referees": [], "ball": []}


    with tqdm(
    total=total_frames,
    desc="Processing video",
    unit="frame",
    dynamic_ncols=True,
    ) as pbar:
        for start_frame in range(0, total_frames, batch_size):
            frame_batch = read_video_batch(video_path, batch_size=batch_size, start_frame=start_frame)
            # print(f"Read {len(frame_batch)} frames from {video_path} starting at frame {start_frame}")

            if not from_stub:
                batch_tracks = tracker.get_object_tracks(frame_batch)
                for key in all_tracks:
                    all_tracks[key].extend(batch_tracks[key])  # frame numbers continue: batch 2 starts at 100

            # Only this batch's tracks, so index 0 = first frame of this batch (not frame 0 of the video)
            batch_tracks = {key: all_tracks[key][start_frame:start_frame + len(frame_batch)] for key in all_tracks}

            # from pprint import pprint
            # for category, frames in all_tracks.items():
            #     print(f"{category}: {len(frames)} frames")

            # pprint(
            #     {category: frames[0] if frames else {}
            #     for category, frames in all_tracks.items()},
            #     sort_dicts=False
            # )
            # return

            output_video_frames = tracker.draw_annotations(frame_batch, batch_tracks)
            writer = save_video(output_video_frames,
                                 output_video_path, 
                                 fps=25, writer=writer)
            batch_count = len(frame_batch)

            pbar.set_postfix(
                last_frame=start_frame + batch_count - 1,
                batch=batch_count,
                refresh=False,
            )
            pbar.update(batch_count)

            
            # print(f"Appended {len(frame_batch)} frames to {output_video_path}")

            del frame_batch,output_video_frames, batch_tracks  # free memory after processing each batch

    writer.release()  # finish the file once, after all batches

    if not from_stub:
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        with open(stub_path, "wb") as f:
            pickle.dump(all_tracks, f)  # saved once, for the whole video
        print(f"Saved tracks for {len(all_tracks['players'])} frames to {stub_path}")


if __name__ == "__main__":
    device = get_device()
    main()
