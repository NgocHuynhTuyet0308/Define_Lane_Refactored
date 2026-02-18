import cv2
import numpy as np

class PerspectiveTransformer:
    """Biến đổi phối cảnh ảnh (perspective transform) giữa góc nhìn gốc và BEV"""

    def __init__(self, image, src_points, dest_points):
        """Khởi tạo với ảnh, 4 điểm nguồn (src) và 4 điểm đích (dest)"""
        self.image = image
        self.src_points = src_points
        self.dest_points = dest_points

    def create_ROI(self):
        """Tạo vùng quan tâm (ROI) bằng cách mask vùng ngoài src_points"""
        polygons = np.array([self.src_points], dtype=np.int32)
        mask = np.zeros_like(self.image)
        cv2.fillPoly(mask, polygons, (255, 255, 255))
        masked_image = cv2.bitwise_and(self.image, mask)
        return masked_image

    def change_perspective(self):
        """Chuyển đổi phối cảnh ảnh từ src_points sang dest_points (warp perspective)"""
        perspective_matrix = cv2.getPerspectiveTransform(self.src_points, self.dest_points)
        height, width = self.image.shape[:2]
        warped_image = cv2.warpPerspective(self.image, perspective_matrix, (width, height))
        return warped_image
