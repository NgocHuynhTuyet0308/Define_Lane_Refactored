import yaml
import os
import argparse
import cv2
from process.lane_segmentation_pipeline import LaneSegmentationPipeline
from tqdm import tqdm


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
    """Xử lý cho từng ảnh"""
    frame = cv2.imread(image_path)

    pipeline = LaneSegmentationPipeline.from_config(config)
    output_image, lane_segment_mask = pipeline.process_frame(frame)

    # In kết quả đầu ra
    cv2.imwrite(os.path.join(output_path, 'output_result.jpg'), output_image)
    cv2.imwrite(os.path.join(output_path, 'lane_segment_mask.jpg'), (lane_segment_mask*255))
    print(f"Output saved to {output_path}")

def process_video(video_path: str, output_path: str, config: dict):
    """Xử lý cho từng video (Frame by frame)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pipeline = LaneSegmentationPipeline.from_config(config)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_result = cv2.VideoWriter(os.path.join(output_path, 'output_result.mp4'), fourcc, fps, (width, height))
    out_lane_segment = cv2.VideoWriter(os.path.join(output_path, 'lane_segment_mask.mp4'), fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        output_frame, lane_segment_mask = pipeline.process_frame(frame)

        out_result.write(output_frame)
        lane_segment_bgr = cv2.cvtColor((lane_segment_mask * 255).astype('uint8'), cv2.COLOR_GRAY2BGR) if len(lane_segment_mask.shape) == 2 else lane_segment_mask
        out_lane_segment.write(lane_segment_bgr)
    
    cap.release()
    out_result.release()
    out_lane_segment.release()
    print(f"Processed video saved to: {output_path}")
        

def main(input_path: str, output_path: str, config_path: str):
    """Hàm chính: load config, phân loại input và gọi hàm xử lý tương ứng"""
    os.makedirs(output_path, exist_ok=True)
    config = load_config(config_path)

    input_type = check_input_type(input_path)

    if input_type == 'image':
        process_image(input_path, output_path, config)
    elif input_type == 'video':
        process_video(input_path, output_path, config)
    else:
        print(f"Unsupported input type: {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate output")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input (Image or Video)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output folder")
    parser.add_argument("--config", type=str, default="data/config/approach_2_config.yaml", help="Path to config.yaml")

    args = parser.parse_args()
    main(args.input_path, args.output_path, args.config)