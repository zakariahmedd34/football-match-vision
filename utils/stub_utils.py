import os
import logging
import pickle
from datetime import datetime

from .logger import log_step


def stub_meta(video_path, model_path):
    """Who made this stub. If any value changes, the stub is out of date."""
    return {
        "video": os.path.basename(video_path),
        "model": os.path.basename(model_path),
        "model_mtime": os.path.getmtime(model_path),   # changes when best.pt is replaced by a new training
    }


def _readable(key, value):
    """model_mtime is a raw timestamp -> show it as a date in the log."""
    if key == "model_mtime" and isinstance(value, (int, float)):
        return f"{datetime.fromtimestamp(value):%Y-%m-%d %H:%M:%S}"
    return value


def _check_stub(stub_path, video_path, model_path, total_frames):
    """Return (tracks or None, reason). Logs every intermediate check."""
    # 1. does the file exist?
    if not os.path.exists(stub_path):
        return None, "no stub file"
    size_kb = os.path.getsize(stub_path) / 1024
    saved_at = datetime.fromtimestamp(os.path.getmtime(stub_path))
    logging.info(f"  stub file found: size={size_kb:.1f} KB | saved_at={saved_at:%Y-%m-%d %H:%M:%S}")

    # 2. can it be read?
    try:
        with open(stub_path, "rb") as f:
            stub = pickle.load(f)
    except Exception as e:
        return None, f"stub file is broken ({type(e).__name__}: {e})"

    # 3. does it have the meta block?
    if not isinstance(stub, dict) or "meta" not in stub:
        keys = list(stub.keys()) if isinstance(stub, dict) else type(stub).__name__
        return None, f"old stub format without meta (contents: {keys})"

    # 4. compare EVERY meta field (not only the first mismatch) so the log shows the full picture
    expected = stub_meta(video_path, model_path)
    mismatches = []
    for key, now in expected.items():
        saved = stub["meta"].get(key)
        status = "OK" if saved == now else "MISMATCH"
        logging.info(f"  meta {key:<12}: stub={_readable(key, saved)} | now={_readable(key, now)} -> {status}")
        if status == "MISMATCH":
            mismatches.append(key)
    if mismatches:
        return None, f"stub was made for a different {', '.join(mismatches)}"

    # 5. enough frames?
    n_frames = len(stub["tracks"]["players"])
    logging.info(f"  frames: stub={n_frames} | needed={total_frames}")
    if n_frames < total_frames:
        return None, f"stub has only {n_frames} frames, need {total_frames}"

    return stub["tracks"], f"stub matches video + model ({n_frames} frames)"


def load_stub(stub_path, video_path, model_path, total_frames):
    """Return saved tracks, or None if the stub is missing, broken, or from another video/model."""
    with log_step("Stub check",
                  stub=stub_path,
                  video=os.path.basename(video_path),
                  model=os.path.basename(model_path),
                  frames_needed=total_frames) as out:
        tracks, reason = _check_stub(stub_path, video_path, model_path, total_frames)
        out["decision"] = "USE STUB (skip detection)" if tracks is not None else "RUN DETECTION"
        out["reason"] = reason
        if tracks is None and os.path.exists(stub_path):
            logging.warning(f"  stub rejected: {reason} -> detection will run and the stub will be overwritten")
        return tracks


def save_stub(stub_path, all_tracks, video_path, model_path):
    """Write to a temp file first, then rename -> a crash mid-write can't leave a broken stub."""
    with log_step("Save stub", stub=stub_path, frames=len(all_tracks["players"])) as out:
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        meta = stub_meta(video_path, model_path)
        tmp_path = stub_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump({"meta": meta, "tracks": all_tracks}, f)
        os.replace(tmp_path, stub_path)
        out["size_kb"] = f"{os.path.getsize(stub_path) / 1024:.1f}"
        out["meta"] = {k: _readable(k, v) for k, v in meta.items()}
