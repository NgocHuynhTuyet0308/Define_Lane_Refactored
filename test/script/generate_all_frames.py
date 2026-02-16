import os
import cv2

input_video_path = 'data\\video\\curved_lane.mp4'
output_folder_path = 'output\\curved_lane_frames'

os.makedirs(os.path.join(output_folder_path), exist_ok=True)

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise ValueError(f"Cannot open video: {input_video_path}")

frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite(os.path.join(output_folder_path, f'frame_{frame_index:04d}.jpg'), frame)
    frame_index += 1

cap.release()
print(f"Processed video saved to: {output_folder_path}")
