from ultralytics import YOLO
import numpy as np
import supervision as sv
import pickle
import os
import sys
import cv2
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path:str) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames:list[np.ndarray]) -> list[list[np.ndarray]]:
        batch_size=30
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1) 
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames: list[np.ndarray], read_from_stub:bool=False, stub_path:str | None=None) -> dict[str, list[dict]]:

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            # Inversion des clé et valeurs de classe ::::: {0: player, 1: ball, ect...} -> {player: 0, ball: 1, ect...}
            class_names = detection.names
            class_names_inv = {v:k for k,v in class_names.items()}

            # Convert to supervision Detection format
            detection_supervision_format = sv.Detections.from_ultralytics(detection)

            # Convert Goalkeeper object to player
            for object_ind, class_id in enumerate(detection_supervision_format.class_id):
                if class_names[class_id] == "goalkeeper":
                    detection_supervision_format.class_id[object_ind] = class_names_inv["player"]

            #Track Objects
            frame_detection_with_tracks = self.tracker.update_with_detections(detection_supervision_format)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for object_detected in frame_detection_with_tracks:
                bbox = object_detected[0].tolist()
                cls_id = object_detected[3] 
                track_id = object_detected[4] 

                if cls_id == class_names_inv["player"] :
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == class_names_inv["referee"] :
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            for object_detected in detection_supervision_format:
                bbox = object_detected[0].tolist()
                cls_id = object_detected[3]

                if cls_id == class_names_inv["ball"]:
                    tracks["ball"][frame_num][0] = {"bbox":bbox}

            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks,f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, 
                    center=(x_center,y2),
                    axes=(int(width), int(0.35*width)), 
                    angle=0.0, 
                    startAngle=-45, 
                    endAngle=235,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4)

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) +15
        y2_rect = (y2 + rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text),int(y1_rect+15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,255,255),
                        2)
            
        return frame

    def draw_traingle_ball(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ], dtype=np.int32)

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame

    def draw_anotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]  
            ball_dict = tracks["ball"][frame_num]  
            
            # Players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0,0,255), track_id)

            # Referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,0,0))     

            # Ball
            for _, ball in ball_dict.items():
                frame = self.draw_traingle_ball(frame, ball["bbox"], (200,50,0)) 
            
            output_video_frames.append(frame)
            
        return output_video_frames