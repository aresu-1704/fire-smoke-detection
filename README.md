  # Fire and Smoke Detection

![Demo](outputs/detected_video.gif)

Hệ thống nhận diện lửa và khói sử dụng YOLOv8 để hỗ trợ phát hiện cháy rừng sớm.

## Tính năng

- Phát hiện lửa (fire) và khói (smoke) trong video/ảnh
- Tối ưu cho xử lý realtime với frame skipping
- Hỗ trợ GPU acceleration (CUDA)
- Màu sắc phân biệt: lửa (đỏ), khói (xám)

## Yêu cầu

```bash
pip install -r requirements.txt
```

## Sử dụng

1. Đặt video input vào thư mục `inputs/`
2. Chạy detection:

```bash
python detect.py
```

## Cấu trúc thư mục

```
fire-smoke-detection/
├── models/          # Mô hình YOLO (.pt)
├── inputs/          # Video/ảnh đầu vào
├── outputs/         # Kết quả detection
├── detect.py        # Script chính
└── requirements.txt
```

## Cấu hình

Chỉnh sửa các tham số trong `detect.py`:

- `CONF_THRESHOLD`: Ngưỡng confidence (mặc định: 0.3)
- `IMG_SIZE`: Kích thước resize (mặc định: 640)
- `FRAME_SKIP`: Số frame bỏ qua (mặc định: 3)

## License

Apache License 2.0
