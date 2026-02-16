import ast
import cv2
import numpy as np

input_path = 'output\\curved_lane_frames\\frame_0000.jpg'
raw = "[500, 475], [800, 475], [1100, 620], [250, 620]"
img = cv2.imread(input_path)

points = [tuple(p) for p in ast.literal_eval(f"[{raw}]")]
print(points)

for i, (x, y) in enumerate(points):
    cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
    cv2.putText(img, f"P{i}({x},{y})", (x+10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
pts = np.array(points, np.int32).reshape((-1, 1, 2))
cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

cv2.imwrite("output/draw_points.jpg", img)
