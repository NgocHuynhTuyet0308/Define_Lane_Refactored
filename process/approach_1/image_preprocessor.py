import numpy as np
import cv2

class ImagePreProcessor:
    """Tiền xử lý ảnh: tạo các mask màu và cạnh để phát hiện làn đường"""

    def __init__(self, undisorted_image: np.ndarray,
                 yellow_lower_bound=(0, 100, 100), yellow_upper_bound=(210, 255, 255),
                 white_threshold=200, white_maxval=255,
                 sobel_k_size=5, sobel_threshold=30
        ):
        """Khởi tạo với ảnh đã khử méo và các tham số lọc màu, Sobel"""
        self.undisorted_image = undisorted_image
        self.yellow_lower_bound = np.array(yellow_lower_bound)
        self.yellow_upper_bound = np.array(yellow_upper_bound)
        self.white_threshold = white_threshold
        self.white_maxval = white_maxval
        self.sobel_k_size = sobel_k_size
        self.sobel_threshold = sobel_threshold

    def create_yellow_color_mask(self):
        """Tạo mask phát hiện làn vàng từ không gian màu HSV"""
        hsv_image = cv2.cvtColor(self.undisorted_image, cv2.COLOR_BGR2HSV)
        yellow_color_mask = cv2.inRange(hsv_image, self.yellow_lower_bound, self.yellow_upper_bound)
        return yellow_color_mask
    
    def create_white_color_mask(self):
        """Tạo mask phát hiện làn trắng bằng ngưỡng grayscale"""
        gray_scale_image = cv2.cvtColor(self.undisorted_image, cv2.COLOR_BGR2GRAY)
        _, mask_white_threshold = cv2.threshold(gray_scale_image, self.white_threshold, self.white_maxval, cv2.THRESH_BINARY)
        return mask_white_threshold

    def create_sobel_edge_mask(self):
        """Tạo mask phát hiện cạnh bằng bộ lọc Sobel"""
        gray_scale_image = cv2.cvtColor(self.undisorted_image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray_scale_image, (self.sobel_k_size, self.sobel_k_size), 0)

        img_sobelx = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=1)
        img_sobely = cv2.Sobel(blurred, cv2.CV_8U, 0, 1, ksize=1)
        img_sobel = (img_sobelx + img_sobely) / 2

        _, sobel_edge_mask = cv2.threshold(img_sobel, self.sobel_threshold, 255, cv2.THRESH_BINARY)

        return sobel_edge_mask



    def preprocess_image(self):
        """Kết hợp các mask (vàng, trắng, Sobel) thành mask cuối cùng"""
        yellow_color_mask = self.create_yellow_color_mask()
        white_color_mask = self.create_white_color_mask()
        sobel_edge_mask = self.create_sobel_edge_mask()

        # Resize image
        height, width = white_color_mask.shape
        sobel_edge_mask = cv2.resize(sobel_edge_mask, (width, height)).astype(np.uint8)
        yellow_color_mask = cv2.resize(yellow_color_mask, (width, height)).astype(np.uint8)
        white_color_mask = white_color_mask.astype(np.uint8)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(sobel_edge_mask, white_color_mask)
        final_mask = cv2.bitwise_or(combined_mask, yellow_color_mask)

        return final_mask



    @classmethod
    def from_config(cls, image: np.ndarray, config: dict):
        """Tạo instance ImagePreProcessor từ ảnh và dict config"""
        return cls(
            undisorted_image=image,
            **config
        )
        
    
