import cv2
import numpy as np

class PerspectiveTransformer:
    def __init__(self, image, src_points, dest_points):
        self.image = image
        self.src_points = src_points
        self.dest_points = dest_points

    def create_ROI(self):
        polygons = np.array([self.src_points], dtype=np.int32)
        mask = np.zeros_like(self.image)
        cv2.fillPoly(mask, polygons, (255, 255, 255))
        masked_image = cv2.bitwise_and(self.image, mask)
        return masked_image

    def change_perspective(self):
        perspective_matrix = cv2.getPerspectiveTransform(self.src_points, self.dest_points)
        height, width = self.image.shape[:2]
        warped_image = cv2.warpPerspective(self.image, perspective_matrix, (width, height))
        return warped_image
