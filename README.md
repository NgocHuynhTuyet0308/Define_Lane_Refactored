# Define_Lane_Refactored

Project: Xây dựng chương trình phát hiện làn đường sử dụng phương pháp học sâu và kỹ thuật xử lý biên ảnh
- **Approach 1**: Xử lý ảnh truyền thống (Camera Calibration + Perspective Transform + Sliding Window)
- **Approach 2**: Deep Learning (U-Net Segmentation)

## Cấu trúc thư mục

```
Define_Lane_Refactored/
├── data/
│   ├── camera_calibration/     # Dữ liệu calibration camera
│   ├── config/                 # File cấu hình YAML
│   │   ├── approach_1_config.yaml
│   │   └── approach_2_config.yaml
│   ├── image/                  # Ảnh test đầu vào mẫu
│   ├── model/                  # Model weight (U-Net)
│   └── video/                  # Video test đầu vào mẫu

├── process/
│   ├── approach_1/             # Module xử lý Approach 1
    ├── approach_2/             # Module xử lý Approach 2
├── test_approach_1.py          # Script chạy Approach 1
├── test_approach_2.py          # Script chạy Approach 2
└── output/                     # Thư mục kết quả đầu ra
```

## 1. Hướng dẫn cài đặt

### Yêu cầu hệ thống
- Python >= 3.8

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## 2. Hướng dẫn chạy code

### Approach 1 - Xử lý ảnh truyền thống

```bash
# Xử lý ảnh
python test_approach_1.py --input_path data/image/0a0379a5-3870b241.jpg --output_path output/approach_1 --config data/config/approach_1_config.yaml --video_type default

# Xử lý video
python test_approach_1.py --input_path data/video/straight_lane.mp4 --output_path output/approach_1 --config data/config/approach_1_config.yaml --video_type straight_lane
```

| Tham số | Mô tả |
|---|---|
| `--input_path` | Đường dẫn tới file ảnh hoặc video đầu vào |
| `--output_path` | Thư mục lưu kết quả |
| `--config` | Đường dẫn file config YAML |
| `--video_type` | Loại video: `straight_lane`, `curved_lane`, `foggy_lane`, `default` (Lưu ý: Do ROI fit cứng nên cần ghi đúng video type tùy theo tên mỗi video test trong folder data\\video) |

### Approach 2 - Deep Learning (U-Net)

```bash
# Xử lý ảnh
python test_approach_2.py --input_path data/image/0a0379a5-3870b241.jpg --output_path output/approach_2

# Xử lý video
python test_approach_2.py --input_path data/video/straight_lane.mp4 --output_path output/approach_2
```

| Tham số | Mô tả |
|---|---|
| `--input_path` | Đường dẫn tới file ảnh hoặc video đầu vào |
| `--output_path` | Thư mục lưu kết quả |
| `--config` | Đường dẫn file config (mặc định: `data/config/approach_2_config.yaml`) |


## 3. Kết quả sau khi thực hiện chạy code

### Approach 1
Kết quả được lưu trong thư mục `output/approach_1/`:

| File | Mô tả |
|---|---|
| `output_result.jpg/mp4` | Ảnh/video gốc với làn đường được vẽ đè |
| `roi.jpg` / `ROI_and_mask.mp4` | Vùng quan tâm (ROI) và mask |
| `wraped_mask.jpg` / `BEV_and_mask.mp4` | Bird's Eye View và mask |

### Approach 2
Kết quả được lưu trong thư mục `output/approach_2/`:

| File | Mô tả |
|---|---|
| `output_result.jpg/mp4` | Ảnh/video gốc với vùng làn đường được tô màu |
| `lane_segment_mask.jpg/mp4` | Mask phân đoạn làn đường |