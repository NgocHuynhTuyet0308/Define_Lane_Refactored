import numpy as np
import pickle
import cv2
import glob
import matplotlib.image as mpimg


class CameraCalibrator:
    def __init__(
        self,
        calibration_file_path: str,
        chessboard_size: tuple = None,
        calibration_images_path: str = "camera_cal/*.jpg"
    ):
        """Khởi tạo Camera Calibrator: load tham số hiệu chỉnh từ file hoặc tính từ ảnh bàn cờ"""
        self.calibration_file_path = calibration_file_path
        self.chessboard_size = chessboard_size if chessboard_size else (9, 6)
        self.calibration_images_path = calibration_images_path
        self.mtx = None
        self.dist = None
        self._load_calibration()

    def _load_calibration(self):
        """Load tham số hiệu chỉnh từ file pickle, nếu không có thì tính từ bộ ảnh bàn cờ"""
        try:
            with open(self.calibration_file_path, "rb") as f:
                dist_pickle = pickle.load(f)
                self.mtx = dist_pickle["mtx"]
                self.dist = dist_pickle["dist"]
        except FileNotFoundError:
            # Fallback: Generate calibration from images
            print(f"Calibration file not found. Generating from images: {self.calibration_images_path}")
            self.mtx, self.dist = self.get_distortion_params(self.calibration_images_path)
            # Save the generated calibration
            dist_pickle = {'mtx': self.mtx, 'dist': self.dist}
            with open(self.calibration_file_path, 'wb') as f:
                pickle.dump(dist_pickle, f)
            print(f"Calibration saved to: {self.calibration_file_path}")
        except Exception as e:
            raise Exception(f"Error loading calibration file: {str(e)}")

    def get_distortion_params(self, calibration_path):
        """Tính toán các tham số hiệu chỉnh camera (mtx, dist) từ bộ ảnh bàn cờ"""
        objp = np.zeros((self.chessboard_size[1] * self.chessboard_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        images = glob.glob(calibration_path)

        if not images:
            raise FileNotFoundError(f"No calibration images found at: {calibration_path}")

        for fname in images:
            img = mpimg.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        if not objpoints:
            raise ValueError("No chessboard corners found in any images")

        img = cv2.imread(images[0])
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        return mtx, dist

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Khử méo ảnh bằng tham số hiệu chỉnh đã load"""
        if self.mtx is None or self.dist is None:
            raise ValueError("Calibration parameters not loaded")

        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    @classmethod
    def from_config(cls, config: dict):
        """Tạo instance CameraCalibrator từ dict config"""
        return cls(
            calibration_file_path=config.get('calibration_file_path'),
            chessboard_size=tuple(config.get('chessboard_size', [9, 6]))
        )