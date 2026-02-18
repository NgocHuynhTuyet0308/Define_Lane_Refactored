import numpy as np
import cv2


class LaneVisualizer:
    """Vẽ và hiển thị kết quả phát hiện làn đường lên ảnh"""

    @staticmethod
    def one_to_three_channel(binary):
        """Chuyển ảnh nhị phân 1 kênh sang ảnh 3 kênh (BGR)"""
        img = np.zeros((binary.shape[0], binary.shape[1], 3), dtype='uint8')
        img[:, :, 0] = binary
        img[:, :, 1] = binary
        img[:, :, 2] = binary
        return img

    @staticmethod
    def draw_curves(img, pts_left, pts_right):
        """Vẽ đường cong làn trái và phải lên ảnh BEV"""
        img = LaneVisualizer.one_to_three_channel(img)
        if pts_left is not None:
            cv2.polylines(img, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=16)
        if pts_right is not None:
            cv2.polylines(img, np.int32([pts_right]), isClosed=False, color=(255, 0, 0), thickness=16)
        return img

    @staticmethod
    def draw_roi_points(image, src_points):
        """Vẽ vùng ROI (4 điểm src_points) lên ảnh gốc"""
        result = image.copy()
        pts = np.int32(src_points)
        cv2.polylines(result, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        return result

    @staticmethod
    def overlay_on_original(original_image, mask_draw_image, inverse_perspective_matrix):
        """Chiếu ngược mask từ BEV về góc nhìn gốc và chồng lên ảnh gốc"""
        inverse_mask = cv2.warpPerspective(
            mask_draw_image, inverse_perspective_matrix,
            (original_image.shape[1], original_image.shape[0])
        )
        result = cv2.addWeighted(original_image, 0.8, inverse_mask, 1, 1)
        return result
