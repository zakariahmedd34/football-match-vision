import os
import cv2
import pickle
import torch
from tqdm import tqdm
from team_assigner import TeamAssigner
from utils.video_utils import read_video_batch, save_video, check_inputs
from utils.logger import logging, log_step
from utils.exception import CustomException
from utils.stub_utils import save_stub,stub_meta,load_stub
import sys
from trackers import Tracker
from player_ball_assignment import PlayerBallAssigner
import numpy as np

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch version: {torch.__version__}")
    print(f"Using device: {device}")
    return device


def main():
    model_path = str(PROJECT_ROOT / "models" / "best.pt")
    video_path = str(PROJECT_ROOT / "input_videos" / "realmadrid.mp4")
    output_video_path = str(PROJECT_ROOT / "output_videos" / "output.mp4")
    stub_path = str(PROJECT_ROOT / "stubs" / "tracks.pkl")


    # crop = True
    total_frames = 750
    batch_size = 100
    writer = None  # created on the first batch, then reused so each batch is appended
    team_ball_control = []  # one entry per VIDEO frame (whole video, not per batch): 1, 2, or 0 = nobody yet

    # Read the real fps from the video (realmadrid.mp4 is 30 fps), so the output plays at normal speed    
    fps, total_frames = check_inputs(video_path, model_path, output_video_path, total_frames)

     # Tracks for the WHOLE video: all_tracks["players"][frame_num] -> {track_id: {"bbox": [...]}}
    # Use the stub only if it matches this video + model; otherwise run detection.
    all_tracks = load_stub(stub_path, video_path, model_path, total_frames)
    from_stub = all_tracks is not None
    if not from_stub:
        all_tracks = {"players": [], "referees": [], "ball": []}

    tracker = Tracker(model_path)  # ONE tracker for all batches -> IDs continue across batches
    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()
    n_batches = (total_frames + batch_size - 1) // batch_size
    logging.info(f"Pipeline config | video={os.path.basename(video_path)} | frames={total_frames} | "
                 f"batch_size={batch_size} | batches={n_batches} | fps={fps:.2f} | "
                 f"tracks_source={'STUB' if from_stub else 'DETECTION'} | output={output_video_path}")

    with tqdm(
    total=total_frames,
    desc="Processing video",
    unit="frame",
    dynamic_ncols=True,
    ) as pbar:
        try:
            for batch_idx, start_frame in enumerate(range(0, total_frames, batch_size), start=1):
                end_frame = min(start_frame + batch_size, total_frames) - 1
                with log_step(f"Batch {batch_idx}/{n_batches}", frames=f"{start_frame}-{end_frame}") as batch_out:

                    frame_batch = read_video_batch(video_path, batch_size=batch_size, start_frame=start_frame)
                    if not frame_batch:
                        logging.warning(f"no frames read at start_frame={start_frame} -> stopping")
                        break

                    if not from_stub:
                        batch_tracks = tracker.get_object_tracks(frame_batch)
                        for key in all_tracks:
                            all_tracks[key].extend(batch_tracks[key])  # frame numbers continue: batch 2 starts at 100
                    else:
                        logging.info("  Detect + track: SKIPPED (tracks taken from stub)")

                    # Only this batch's tracks, so index 0 = first frame of this batch (not frame 0 of the video)
                    batch_tracks = {key: all_tracks[key][start_frame:start_frame + len(frame_batch)] for key in all_tracks}

                    batch_tracks['ball'] = tracker.interpolate_ball_positions(batch_tracks["ball"])

                    # Fit the 2 team colors ONCE (first batch), otherwise
                    # KMeans may swap team 1/2 between batches and the colors would flip.
                    if not team_assigner.team_colors:
                        team_assigner.assign_team_color(frame_batch[0], batch_tracks['players'][0])
                    team_assigner.assign_teams(frame_batch, batch_tracks['players'])

                    # OLD version (no 5-frame rule): appends one entry per frame to team_ball_control
                    player_assigner.assign_ball_for_batch(batch_tracks['players'], batch_tracks['ball'],
                                                          team_ball_control)

                    output_video_frames = tracker.draw_annotations(frame_batch, batch_tracks,
                                                                   np.array(team_ball_control), start_frame)
                    writer = save_video(output_video_frames, output_video_path,
                                        fps=int(fps), writer=writer)
                    batch_count = len(frame_batch)
                    batch_out["frames_done"] = f"{start_frame + batch_count}/{total_frames}"

                pbar.set_postfix(
                    last_frame=start_frame + batch_count - 1,
                    batch=batch_count,
                    refresh=False,
                )
                pbar.update(batch_count)

                del frame_batch,output_video_frames, batch_tracks  # free memory after processing each batch

        except Exception as e:

            error = CustomException(e, sys)
            logging.error(error)          # write it to the log file, not only the terminal
            raise error from e
        finally:
            if writer is not None:
                writer.release()          # close the mp4 even after a crash, so it isn't corrupted
    

    if not from_stub:
        save_stub(stub_path, all_tracks, video_path, model_path)
    else:
        logging.info("Save stub: SKIPPED (tracks came from a valid stub)")

if __name__ == "__main__":
    device = get_device()
    with log_step("Pipeline", device=device):
        main()
