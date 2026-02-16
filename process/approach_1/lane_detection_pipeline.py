import numpy as np
import cv2
from .camera_calibration import CameraCalibrator
from .image_preprocessor import ImagePreProcessor
from .perspective_transformer import PerspectiveTransformer
from .lane_detector import LaneDetector
from .lane_visualizer import LaneVisualizer


class LaneDetectionPipeline:
    """Pipeline for processing frames through all lane detection steps"""

    def __init__(self, config: dict):
        """
        Initialize pipeline with configuration

        Args:
            config: Dictionary containing all configuration parameters
        """
        self.config = config

        # Initialize camera calibrator
        self.calibrator = CameraCalibrator.from_config(config['camera_calibration'])

        # Store preprocessing config
        self.preprocessing_config = config.get('image_preprocessing', {})

        # Store perspective transform config
        pt_config = config.get('perspective_transform', {})
        video_type = config.get('video_type', 'straight_lane')
        self.perspective_config = pt_config.get(video_type, {})

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the entire pipeline

        Args:
            frame: Input frame (BGR image)

        Returns:
            Processed frame
        """
        # Step 1: Undistort image
        undistorted = self.calibrator.undistort_image(frame)

        # Step 2: Apply image preprocessing (yellow lane detection)
        preprocessor = ImagePreProcessor(
            undisorted_image=undistorted,
            k_size_sobel_filter=self.preprocessing_config.get('k_size_sobel_filter', 5),
            threshold_sobel_filter=self.preprocessing_config.get('threshold_sobel_filter', 30),
            yellow_lower_bound=self.preprocessing_config.get('yellow_lower_bound', [0, 100, 100]),
            yellow_upper_bound=self.preprocessing_config.get('yellow_upper_bound', [210, 255, 255]),
            white_threshold = self.preprocessing_config.get('white_threshold', 200),
            white_maxval = self.preprocessing_config.get('white_maxval', 255),
        )
        combined_mask = preprocessor.preprocess_image()
    

        # TODO: Add more processing steps here
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

        # Step 4: Lane detection
        kernel = np.ones((11, 11), np.uint8)
        bird_eye_image_morphology = cv2.morphologyEx(wraped_combined_mask, cv2.MORPH_CLOSE, kernel)
        left_fit, right_fit = LaneDetector.fit_curve(bird_eye_image_morphology)
        pts_left, pts_right = LaneDetector.find_points((960, 400), left_fit, right_fit)

        # Step 5: Draw lanes on original image
        mask_draw_image = LaneVisualizer.draw_curves(wraped_combined_mask, pts_left, pts_right)
        original_image = LaneVisualizer.overlay_on_original(
            undistorted, mask_draw_image, inverse_perspective_matrix
        )

        return original_image

    @classmethod
    def from_config(cls, config: dict):
        """Create pipeline from configuration dictionary"""
        return cls(config)
