from ultralytics import YOLO
import supervision as sv
import cv2
from cv2 import ellipse
import sys
import numpy as np
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()  # Initialize the tracker (keeps its state between calls -> IDs continue across batches)

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            results = self.model.predict(batch_frames, conf = 0.1, verbose=False)
            detections.extend(results)
        return detections

    def get_object_tracks(self, frames):
        """Detect + track one batch of frames. Call it with batches in video order."""

        detections = self.detect_frames(frames)

        tracks = {

            "players": [],
            "referees": [],
            "ball": []

        }


        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {V:k for k, V in cls_names.items()}

            # convet the detection results to supervision Detections format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goalkeeper to a plyaer object
            for object_index, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_index] = cls_names_inv['player']


            # Track Objects

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                ## extract the bb
                bbox = frame_detection[0].tolist()  # convert to list
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {'bbox': bbox}

                elif cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {'bbox': bbox}

            for frame_detection in detection_supervision:
                ## extract the bb
                bbox = frame_detection[0].tolist()  # convert to list
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {'bbox': bbox}

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):

        # botton
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(
        frame,
        center=(int(x_center), y2),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4)

        rectangle_width = 40
        rectangle_hight = 20

        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2-rectangle_hight//2) + 15
        y2_rect = (y2+ rectangle_hight//2) + 2

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect),int(y1_rect)),
                (int(x2_rect),int(y2_rect)),
                color,
                cv2.FILLED
            )

            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame


    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _= get_center_of_bbox(bbox)

        triangle_points = np.array([

            [x,y],
            [x-10,y-20],
            [x+10,y-20]

        ])
        cv2.drawContours(frame, [triangle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame



    def draw_annotations(self,video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))  # team color, or red if no team assigned
                frame = self.draw_ellipse(frame=frame,
                                          bbox=player["bbox"],
                                          color=tuple(int(c) for c in color),  # KMeans gives floats; OpenCV wants ints
                                          track_id=track_id)
            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame=frame,
                                            bbox=referee["bbox"],
                                            color=(0,255,255)
                                            )


            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame=frame,
                                           bbox=ball["bbox"],
                                           color = (0,255,0))

            output_video_frames.append(frame)

        return output_video_frames