import argparse
import os
import yaml
import cv2
from tqdm import tqdm
from process.approach_1.lane_detection_pipeline import LaneDetectionPipeline

def load_config(config_path: str) -> dict:
    """Load config từ file YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_input_type(input_path: str):
    """Kiểm tra loại input (image/video) dựa trên phần mở rộng file"""
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    ext = os.path.splitext(input_path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    elif ext in VIDEO_EXTS:
        return "video"
    else:
        return "unknown"


def process_image(image_path: str, output_path: str, config: dict):
    """Xử lý cho từng ảnh: chạy pipeline và lưu output_result, ROI, BEV"""
    os.makedirs(output_path, exist_ok=True)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.resize(image, (1280, 720))

    # Initialize pipeline
    pipeline = LaneDetectionPipeline.from_config(config)

    # Process frame through pipeline
    result, roi_and_mask, bev_and_mask = pipeline.process_frame(image)

    # Save result
    cv2.imwrite(os.path.join(output_path, 'output_result.jpg'), result)
    cv2.imwrite(os.path.join(output_path, 'roi.jpg'), roi_and_mask)
    cv2.imwrite(os.path.join(output_path, 'wraped_mask.jpg'), bev_and_mask)
    print(f"Processed image saved to: {output_path}")


def process_video(video_path: str, output_path: str, config: dict):
    """Xử lý cho từng video (Frame by frame): chạy pipeline và lưu output_result, ROI, BEV"""
    os.makedirs(output_path, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize pipeline
    pipeline = LaneDetectionPipeline.from_config(config)

    # Initialize video writers after first frame to get correct output sizes
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_result = None
    out_roi = None
    out_bev = None

    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))

        # Process frame through pipeline
        result, roi_and_mask, bev_and_mask = pipeline.process_frame(frame)

        # Convert grayscale to BGR if needed (for video output)
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        # Create video writers on first frame with actual output sizes
        if out_result is None:
            h_r, w_r = result.shape[:2]
            h_roi, w_roi = roi_and_mask.shape[:2]
            h_bev, w_bev = bev_and_mask.shape[:2]
            out_result = cv2.VideoWriter(os.path.join(output_path, 'output_result.mp4'), fourcc, fps, (w_r, h_r))
            out_roi = cv2.VideoWriter(os.path.join(output_path, 'ROI_and_mask.mp4'), fourcc, fps, (w_roi, h_roi))
            out_bev = cv2.VideoWriter(os.path.join(output_path, 'BEV_and_mask.mp4'), fourcc, fps, (w_bev, h_bev))

        out_result.write(result)
        out_roi.write(roi_and_mask)
        out_bev.write(bev_and_mask)


    cap.release()
    out_result.release()
    out_roi.release()
    out_bev.release()
    print(f"Processed video saved to: {output_path}")


def main(input_path: str, output_path: str, config_path: str, video_type: str):
    """Hàm chính: load config, phân loại input và gọi hàm xử lý tương ứng"""
    config = load_config(config_path)
    config['video_type'] = video_type

    # Check input type
    input_type = check_input_type(input_path)

    if input_type == 'image':
        process_image(input_path, output_path, config)
    elif input_type == 'video':
        process_video(input_path, output_path, config)
    else:
        raise ValueError(f"Unsupported file type: {input_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate output")
    parser.add_argument("--input_path", type=str, help="Path to input (Image or Video)")
    parser.add_argument("--output_path", type=str, help="Path to output folder")
    parser.add_argument("--config", type=str, help="Path to file config.yaml")
    parser.add_argument("--video_type", type=str, default="default", help="Video type (e.g. straight_lane, curved_lane, foggy_lane or default)")

    args = parser.parse_args()
    main(args.input_path, args.output_path, args.config, args.video_type)