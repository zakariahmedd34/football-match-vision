import os
import pickle
import torch
from tqdm import tqdm
from team_assigner import TeamAssigner
from utils.video_utils import read_video, read_video_batch, save_video
import cv2
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

    # crop = True
    total_frames = 750
    batch_size = 100
    writer = None  # created on the first batch, then reused so each batch is appended

    # Tracks for the WHOLE video: all_tracks["players"][frame_num] -> {track_id: {"bbox": [...]}}
    # If a stub exists from a previous run, load it once and skip detection.
    tracker = Tracker(model_path)  # ONE tracker for all batches -> IDs continue across batches
    team_assigner = TeamAssigner()
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

            if not from_stub:
                batch_tracks = tracker.get_object_tracks(frame_batch)
                for key in all_tracks:
                    all_tracks[key].extend(batch_tracks[key])  # frame numbers continue: batch 2 starts at 100

            # Only this batch's tracks, so index 0 = first frame of this batch (not frame 0 of the video)
            batch_tracks = {key: all_tracks[key][start_frame:start_frame + len(frame_batch)] for key in all_tracks}

            # if crop:
            #     # save cropped image of a player
            #     for track_id, player in batch_tracks['players'][0].items():
            #         bbox = player["bbox"]
            #         frame = frame_batch[0]
            #         # crop bbox from frame
            #         cropped_image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
            #         cv2.imwrite(f'output_videos/cropped_img.jpg',cropped_image) 
            #         crop = False       
            #         break

            # Assign Player Teams: fit the 2 team colors ONCE (first batch), otherwise
            # KMeans may swap team 1/2 between batches and the colors would flip.
            if not team_assigner.team_colors:
                team_assigner.assign_team_color(frame_batch[0],
                                                batch_tracks['players'][0])


            for frame_num, player_track in enumerate(batch_tracks['players']):
                for player_id, track in player_track.items():
                    team = team_assigner.get_player_team(frame_batch[frame_num],
                                                         track['bbox'],
                                                         player_id)
                    batch_tracks['players'][frame_num][player_id]['team'] = team
                    batch_tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

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
