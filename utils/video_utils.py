import cv2
import numpy as np

def read_video(input_video_path:str) -> list[np.ndarray]:
    capture = cv2.VideoCapture(input_video_path)
    frames = []
    while True:
        ret, frame = capture.read()
        if not ret : 
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames:list[np.ndarray], output_video_path:str) -> None:
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    height, width, layers = output_video_frames[0].shape
    out_video = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    for frame in output_video_frames:
        out_video.write(frame)
    out_video.release()