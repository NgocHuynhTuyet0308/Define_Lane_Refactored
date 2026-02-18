import numpy as np
import tensorflow as tf


class LaneSegmentationPipeline:
    """Pipeline xử lý phân đoạn làn đường từ ảnh/video"""

    def __init__(self, config: dict):
        """Khởi tạo pipeline: load model và các tham số từ config"""
        model_config = config["model"]
        self.image_size = model_config["image_size"]
        self.threshold = model_config["threshold"]
        self.model = tf.keras.models.load_model(model_config["weight_path"])

    def preprocess_frame(self, frame: np.ndarray):
        """Tiền xử lý frame: resize, chuẩn hóa và thêm batch dimension"""
        frame = tf.convert_to_tensor(frame, dtype=tf.float32)
        frame = tf.image.resize(frame, self.image_size)
        frame = frame / 255.0
        frame = tf.expand_dims(frame, axis=0)
        return frame

    def predict_mask(self, preprocessed_frame):
        """Dự đoán mask phân đoạn làn đường từ frame đã tiền xử lý"""
        predicted_mask = self.model.predict(preprocessed_frame)[0]
        predicted_mask = np.squeeze(predicted_mask, axis=-1)
        mask = (predicted_mask < self.threshold).astype(np.uint8)
        return mask

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Xử lý 1 frame: tiền xử lý, dự đoán mask và vẽ segment lên ảnh gốc"""
        original_h, original_w = frame.shape[:2]
        preprocessed_frame = self.preprocess_frame(frame)
        mask = self.predict_mask(preprocessed_frame)

        # Resize mask về kích thước gốc
        mask_resized = tf.image.resize(
            mask[..., np.newaxis], [original_h, original_w], method="nearest"
        )
        mask_resized = np.squeeze(mask_resized.numpy(), axis=-1).astype(np.uint8)

        # Vẽ segment lên ảnh RGB gốc
        output_image = frame.copy()
        output_image[mask_resized == 1] = [255, 0, 0]

        return output_image, mask_resized

    @classmethod
    def from_config(cls, config: dict):
        """Tạo instance pipeline từ dict config"""
        return cls(config)

