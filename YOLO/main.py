!pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO
from google.colab import files
from IPython.display import Video, display

# Upload video
uploaded = files.upload()

# Get filename (1155.mp4)
video_path = list(uploaded.keys())[0]
print("Using video:", video_path)

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

# Video properties
# Gets how many frames per second do we have. 
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
output_path = "annotated_1155.mp4"
#Format that it is Mp4
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

#Loops. Gets frame per second and tells us if it works. 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Draw boxes
    annotated = results[0].plot()

    out.write(annotated)

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

cap.release()
out.release()

print("Done!")

# Show video
print("Video saved:", output_path)

from google.colab import files
files.download("annotated_1155.mp4")
