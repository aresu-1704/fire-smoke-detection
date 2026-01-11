"""
Script phát hiện drone sử dụng Ultralytics YOLO
Tối ưu cho realtime với frame skipping và resize
"""
from ultralytics import YOLO
import cv2
from pathlib import Path
import time
import torch


def find_model_file(models_dir="models"):
    """
    Tự động tìm file .pt trong thư mục models
    
    Args:
        models_dir: Đường dẫn thư mục chứa models
        
    Returns:
        Đường dẫn đến file .pt đầu tiên tìm thấy
    """
    models_path = Path(models_dir)
    pt_files = list(models_path.glob("*.pt"))
    
    if not pt_files:
        raise FileNotFoundError(f"Không tìm thấy file .pt trong thư mục {models_dir}")

    model_file = pt_files[0]
    return str(model_file)


def detect_fire_video(model_path, video_path, output_path=None, conf_threshold=0.25, 
                       img_size=640, frame_skip=3, device=None):
    """
    Phát hiện cháy trong video với frame skipping và resize
    
    Args:
        model_path: Đường dẫn đến model YOLO (.pt)
        video_path: Đường dẫn video input
        output_path: Đường dẫn lưu video kết quả (optional)
        conf_threshold: Ngưỡng confidence để hiển thị detection
        img_size: Kích thước resize cho detection (mặc định 640x640)
        frame_skip: Số frame skip giữa mỗi lần detect (mặc định 3 = detect mỗi 4 frame)
        device: 'cuda' hoặc 'cpu' (auto-detect nếu None)
    """
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = YOLO(model_path)
    model.to(device)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")
    
    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tạo VideoWriter nếu cần lưu output
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detect_count = 0
    last_results = None
    
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chỉ detect mỗi (frame_skip + 1) frame
        if frame_count % (frame_skip + 1) == 0:
            # Resize frame về img_size x img_size cho detection
            resized_frame = cv2.resize(frame, (img_size, img_size))
            
            # Chạy detection trên frame đã resize
            results = model(resized_frame, conf=conf_threshold, verbose=False, device=device)
            last_results = results[0]
            detect_count += 1
        
        # Vẽ kết quả lên frame gốc (nếu có detection)
        if last_results is not None:
            # Scale bounding boxes từ 640x640 về kích thước gốc
            boxes = last_results.boxes
            annotated_frame = frame.copy()
            
            for box in boxes:
                # Lấy tọa độ và scale về kích thước gốc
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1 = int(x1 * width / img_size)
                y1 = int(y1 * height / img_size)
                x2 = int(x2 * width / img_size)
                y2 = int(y2 * height / img_size)
                
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                label = f"{class_name} {conf:.2f}"
                
                # Chọn màu theo class: fire = đỏ, smoke = xám
                if class_name.lower() == 'fire':
                    color = (0, 0, 255)  # Đỏ
                elif class_name.lower() == 'smoke':
                    color = (128, 128, 128)  # Xám
                else:
                    color = (0, 255, 0)  # Xanh lá (mặc định)
                
                # Vẽ bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            annotated_frame = frame
        
        # Thêm FPS counter
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Lưu hoặc hiển thị
        if out:
            out.write(annotated_frame)
        
        cv2.imshow('Drone Detection - Realtime', annotated_frame)
        
        frame_count += 1
        
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time

def detect_fire_image(model_path, image_path, output_path=None, conf_threshold=0.25, img_size=640):
    """
    Phát hiện cháy trong ảnh với resize
    
    Args:
        model_path: Đường dẫn đến model YOLO (.pt)
        image_path: Đường dẫn ảnh input
        output_path: Đường dẫn lưu ảnh kết quả (optional)
        conf_threshold: Ngưỡng confidence
        img_size: Kích thước resize cho detection
    """
    
    # Load model
    model = YOLO(model_path)
    
    # Đọc ảnh gốc
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    orig_h, orig_w = original_img.shape[:2]
    
    # Resize về img_size x img_size cho detection
    resized_img = cv2.resize(original_img, (img_size, img_size))
    
    # Chạy detection
    results = model(resized_img, conf=conf_threshold)
    
    # Vẽ kết quả lên ảnh gốc
    annotated_image = original_img.copy()
    boxes = results[0].boxes
    
    for box in boxes:
        # Scale bounding boxes về kích thước gốc
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1 = int(x1 * orig_w / img_size)
        y1 = int(y1 * orig_h / img_size)
        x2 = int(x2 * orig_w / img_size)
        y2 = int(y2 * orig_h / img_size)
        
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = model.names[cls]
        label = f"{class_name} {conf:.2f}"
        
        # Chọn màu theo class: fire = đỏ, smoke = xám
        if class_name.lower() == 'fire':
            color = (0, 0, 255)  # Đỏ
        elif class_name.lower() == 'smoke':
            color = (128, 128, 128)  # Xám
        else:
            color = (0, 255, 0)  # Xanh lá (mặc định)
        
        # Vẽ bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Lưu hoặc hiển thị
    if output_path:
        cv2.imwrite(output_path, annotated_image)
    
    cv2.imshow('Fire-smoke Detection', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Tự động tìm model .pt trong thư mục models
    try:
        MODEL_PATH = find_model_file("models")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[INFO] Vui lòng đặt file .pt vào thư mục models/")
        exit(1)
    
    # Cấu hình
    VIDEO_PATH = "inputs/example.mp4"
    OUTPUT_PATH = "outputs/detected_video.mp4"
    CONF_THRESHOLD = 0.3
    IMG_SIZE = 640
    FRAME_SKIP = 3
    
    # Phát hiện trong video
    try:
        detect_fire_video(
            model_path=MODEL_PATH,
            video_path=VIDEO_PATH,
            output_path=OUTPUT_PATH,
            conf_threshold=CONF_THRESHOLD,
            img_size=IMG_SIZE,
            frame_skip=FRAME_SKIP,
            device='cpu'
        )
    except Exception as e:
        print(f"\n[ERROR] {e}")