from ultralytics import YOLO
import os

model = YOLO('models/bestV5.pt')

results = model.predict('input_videos/video.mp4', save=True)

print(results[0])
print("===============================================")
for box in results[0].boxes:
    print(box)

