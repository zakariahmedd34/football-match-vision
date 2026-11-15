# Tracker output (`tracks`): what's inside

All the examples below are **real values** from `stubs/tracks.pkl` (the 750-frame `full.mp4` video).

---

## 1. The most important point: tracks hold no images

`tracks` contains **only numbers**: box coordinates and IDs. The video frames (the images) are not in it.

| What | Where it lives | Size |
|---|---|---|
| Images (pixels) | `frame_batch`: only the **current batch**, read from the video file each loop | 100 frames ≈ 620 MB |
| Tracks (numbers) | `all_tracks`: the **whole video**, kept for the entire run | 750 frames ≈ 1–2 MB |

So in each loop iteration you have **100 images** (`frame_batch`) but **750 frames of tracks** (`all_tracks`).
Joining these two correctly is what the slicing in section 4 does.

---

## 2. The shape, level by level

```
all_tracks                          -> dict with 3 keys
 ├── "players"                      -> list, one item per video frame (750 items)
 │     ├── [0]                      -> dict for frame 0:  { track_id: {"bbox": [...]}, ... }
 │     ├── [1]                      -> dict for frame 1
 │     ├── ...
 │     └── [749]                    -> dict for frame 749
 ├── "referees"                     -> same shape as "players" (750 items)
 └── "ball"                         -> same shape, but the key is always 1 (there is only one ball)
```

To get one box, read the path from left to right:

```python
all_tracks["players"][100][1]["bbox"]
#           ^ object   ^ frame ^ track ID ^ box
# -> [1073.5, 632.1, 1112.6, 717.1]   (player #1 in video frame 100)
```

| Level | Type | Meaning |
|---|---|---|
| `all_tracks["players"]` | `list` | one entry **per frame** |
| `[100]` | list index | the **frame number in the video** |
| `[1]` | dict key | the **track ID** (the same player keeps the same ID across frames) |
| `["bbox"]` | dict key | the box `[x1, y1, x2, y2]` in pixels: top-left corner, bottom-right corner |

---

## 3. A real sample: video frame 0

```python
{
  "players": [                                   # list of 750 frames
    {                                            # <- index 0 = video frame 0 (21 players)
      1:  {"bbox": [851.9, 633.9, 900.0, 721.2]},
      2:  {"bbox": [1571.2, 611.3, 1612.7, 695.2]},
      3:  {...},
      # ... 21 players in total
    },
    { 1: {"bbox": [851.9, 633.9, 900.0, 721.3]}, ... },   # <- index 1 = video frame 1
    ...                                                   # ... up to index 749
  ],

  "referees": [
    {                                            # video frame 0 (3 referees)
      14: {"bbox": [1853.2, 808.5, 1892.8, 918.3]},
      17: {"bbox": [309.4, 227.0, 327.8, 271.7]},
      20: {"bbox": [776.3, 367.8, 801.8, 425.7]},
    },
    ...
  ],

  "ball": [
    { 1: {"bbox": [1186.9, 848.6, 1203.3, 866.7]} },   # video frame 0: ball found
    ...
    {},                                                # video frame 25: ball NOT found -> empty dict
    ...
  ],
}
```

### Following one player through time

Player #1 has the **same ID** in every frame, so you can follow them:

| Video frame | `all_tracks["players"][frame][1]["bbox"]` |
|---|---|
| 0 | `[851.9, 633.9, 900.0, 721.2]` |
| 1 | `[851.9, 633.9, 900.0, 721.3]` |
| 2 | `[857.6, 636.9, 905.2, 724.4]` |
| 99 | `[1072.4, 634.9, 1113.4, 717.4]` ← last frame of batch 1 |
| 100 | `[1073.5, 632.1, 1112.6, 717.1]` ← first frame of batch 2: same ID, box barely moved |
| 199 | `[1166.3, 536.6, 1191.1, 597.4]` |
| 200 | `[1166.6, 535.5, 1191.7, 596.5]` |

Frames 99 → 100 cross a batch boundary, and the ID continues. That's because one `Tracker` object (one ByteTrack)
is used for all batches.

---

## 4. How a batch joins up with `all_tracks` (the `start:start+batch` slice)

Take **batch 2** (`start_frame = 100`, `batch_size = 100`):

```
video frame number:     100   101   102  ...  199
                         │     │     │         │
frame_batch[i]:          [0]   [1]   [2]  ...  [99]      <- images: counted from 0 inside the batch
all_tracks[...][?]:      [100] [101] [102] ... [199]     <- tracks: counted by VIDEO frame number
```

The two are counted differently:

- `frame_batch` **restarts at 0** in every batch.
- `all_tracks` counts the **whole video**: 0 … 749.

So image `frame_batch[0]` matches `all_tracks["players"][100]`, **not** `all_tracks["players"][0]`.

**General rule:** `frame_batch[i]` ⟷ `all_tracks[...][start_frame + i]`

### What the slice does

```python
batch_tracks = {key: all_tracks[key][start_frame:start_frame + len(frame_batch)] for key in all_tracks}
```

It cuts out only this batch's frames and **renumbers them from 0**, so they line up with `frame_batch`:

| `i` (inside the batch) | `frame_batch[i]` | `batch_tracks["players"][i]` | is the same as |
|---|---|---|---|
| 0 | image of video frame 100 | tracks of video frame 100 | `all_tracks["players"][100]` |
| 1 | image of video frame 101 | tracks of video frame 101 | `all_tracks["players"][101]` |
| 99 | image of video frame 199 | tracks of video frame 199 | `all_tracks["players"][199]` |

Now `draw_annotations(frame_batch, batch_tracks)` can safely use the same `frame_num` for both:

```python
for frame_num, frame in enumerate(video_frames):        # frame_num = 0, 1, 2 ... 99
    player_dict = tracks["players"][frame_num]          # batch_tracks[...][0] = video frame 100 ✔
```

### What went wrong before the slice

`draw_annotations(frame_batch, all_tracks)` used `frame_num` 0…99 directly on `all_tracks`:

```
image of video frame 100  +  tracks of video frame 0    ✘  (ellipses from 4 seconds earlier)
image of video frame 101  +  tracks of video frame 1    ✘
```

Every batch was drawn with the tracks of frames 0–99.

> The same trap applies when printing. `all_tracks["players"][0]` is **always video frame 0**, whichever
> batch you're in. To print the current batch's first frame, use `batch_tracks["players"][0]`
> or `all_tracks["players"][start_frame]`.

---

## 5. Small details

| Detail | Example |
|---|---|
| **Ball key is always `1`** (there is only one ball, so it isn't tracked) | `all_tracks["ball"][0][1]["bbox"]` |
| **No ball found → empty dict `{}`**, so check before reading | 177 of 750 frames have no ball (the first is frame 25). Use `if 1 in all_tracks["ball"][f]:` |
| **Goalkeepers are inside `"players"`** (converted in `get_object_tracks`) | — |
| **Track IDs are numpy integers** (`np.int64(1)`), but `[1]` still works as a key | `all_tracks["players"][0][1]` ✔ |
| **IDs can jump high** (e.g. referee `184` and `236` at frame 450): ByteTrack gives a new ID when it loses someone | — |
| **Length of every list = number of frames processed** | `len(all_tracks["players"]) == 750` |

---

## 6. Quick reference

```python
f = 450                                   # a video frame number

all_tracks["players"][f]                  # all players in frame 450 -> {track_id: {"bbox": [...]}}
len(all_tracks["players"][f])             # how many players -> 20
all_tracks["players"][f][2]["bbox"]       # box of player #2 in frame 450
all_tracks["ball"][f].get(1)              # ball box, or None if not detected

for track_id, obj in all_tracks["referees"][f].items():
    x1, y1, x2, y2 = obj["bbox"]          # every referee in frame 450

# inside the batch loop:
frame_batch[i]                            # image of video frame  start_frame + i
batch_tracks["players"][i]                # tracks of video frame start_frame + i
```
