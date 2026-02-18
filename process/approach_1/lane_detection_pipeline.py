import numpy as np
import cv2
from .camera_calibration import CameraCalibrator
from .image_preprocessor import ImagePreProcessor
from .perspective_transformer import PerspectiveTransformer
from .lane_detector import LaneDetector
from .lane_visualizer import LaneVisualizer


class LaneDetectionPipeline:
    """Pipeline phát hiện làn đường từ ảnh/video"""

    def __init__(self, config: dict):
        """Khởi tạo pipeline: load calibrator và các tham số từ config"""
        self.config = config

        # Initialize camera calibrator
        self.calibrator = CameraCalibrator.from_config(config['camera_calibration'])

        # Store preprocessing config
        self.preprocessing_config = config.get('image_preprocessing', {})

        # Store lane detection config
        self.lane_detection_config = config.get('lane_detection', {})

        # Store perspective transform config
        pt_config = config.get('perspective_transform', {})
        video_type = config.get('video_type', 'straight_lane')
        self.perspective_config = pt_config.get(video_type, {})

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Xử lý 1 frame qua toàn bộ pipeline: undistort, preprocess, perspective transform, detect lane và vẽ kết quả"""
        # Step 1: Undistort image
        undistorted = self.calibrator.undistort_image(frame)

        # Step 2: Apply image preprocessing (yellow lane detection)
        preprocessor = ImagePreProcessor.from_config(undistorted, self.preprocessing_config)
        combined_mask = preprocessor.preprocess_image()
    
        # Step 3: Perspective transform
        perspective_transfomer = PerspectiveTransformer(
            image=combined_mask,
            src_points=np.float32(self.perspective_config['src_points']),
            dest_points=np.float32(self.perspective_config['dest_points'])
        )
        combined_mask_ROI = perspective_transfomer.create_ROI()
        perspective_transfomer.image = combined_mask_ROI
        wraped_combined_mask = perspective_transfomer.change_perspective()
        inverse_perspective_matrix = cv2.getPerspectiveTransform(
            perspective_transfomer.dest_points, perspective_transfomer.src_points
        )

        # BEV of RGB image
        rgb_transformer = PerspectiveTransformer(
            image=undistorted,
            src_points=perspective_transfomer.src_points,
            dest_points=perspective_transfomer.dest_points
        )
        rgb_ROI = rgb_transformer.create_ROI()
        rgb_transformer.image = rgb_ROI
        wraped_rgb = rgb_transformer.change_perspective()

        # Step 4: Lane detection
        kernel_size = self.lane_detection_config.get('morphology_kernel_size', 11)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        bird_eye_image_morphology = cv2.morphologyEx(wraped_combined_mask, cv2.MORPH_CLOSE, kernel)
        left_fit, right_fit = LaneDetector.fit_curve(
            bird_eye_image_morphology,
            nwindows=self.lane_detection_config.get('nwindows', 100),
            margin=self.lane_detection_config.get('margin', 100),
            minpix=self.lane_detection_config.get('minpix', 50),
        )
        img_shape = wraped_combined_mask.shape[:2]
        pts_left, pts_right = LaneDetector.find_points(img_shape, left_fit, right_fit)

        # Step 5: Draw lanes on original image
        mask_draw_image = LaneVisualizer.draw_curves(wraped_combined_mask, pts_left, pts_right)
        original_image = LaneVisualizer.overlay_on_original(
            undistorted, mask_draw_image, inverse_perspective_matrix
        )
        roi_image = LaneVisualizer.draw_roi_points(undistorted, self.perspective_config['src_points'])
    
        # Concat roi_image and combined_mask_ROI
        combined_mask_ROI_bgr = cv2.cvtColor(combined_mask_ROI, cv2.COLOR_GRAY2BGR)
        roi_and_mask = cv2.hconcat([roi_image, combined_mask_ROI_bgr])

        # Concat BEV RGB and BEV mask
        wraped_combined_mask_bgr = cv2.cvtColor(wraped_combined_mask, cv2.COLOR_GRAY2BGR)
        bev_and_mask = cv2.hconcat([wraped_rgb, wraped_combined_mask_bgr])

        return original_image, roi_and_mask, bev_and_mask
    
    @classmethod
    def from_config(cls, config: dict):
        """Tạo instance pipeline từ dict config"""
        return cls(config)
