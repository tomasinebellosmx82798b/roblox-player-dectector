import os
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO

def setup_flexible_paths():
    """
    Tự động tìm kiếm thư mục dữ liệu và cấu hình đường dẫn linh hoạt.
    """
    current_dir = Path.cwd()
    print(f"--- Kiểm tra hệ điều hành: {os.name} ({'Windows' if os.name == 'nt' else 'Linux'}) ---")
    print(f"Thư mục hiện tại: {current_dir}")

    # Danh sách các thư mục ảnh tiềm năng (phẳng hoặc phân cấp)
    # Thứ tự ưu tiên: cấu hình chuẩn YOLO -> thư mục roblox_data -> thư mục images
    search_patterns = [
        "images/train",
        "roblox_esp_ai_legit/images/train",
        "roblox_data",
        "images/roblox_data",
        "images"
    ]

    img_path = None
    for pattern in search_patterns:
        test_path = current_dir / pattern
        if test_path.exists() and any(test_path.iterdir()):
            img_path = test_path
            break

    if not img_path:
        print("LỖI: Không tìm thấy thư mục chứa ảnh!")
        return None

    print(f"Dữ liệu ảnh xác nhận tại: {img_path}")
    
    # Xóa cache để tránh lỗi nhãn cũ
    for cache in current_dir.rglob("*.cache"):
        try:
            cache.unlink()
        except:
            pass

    return img_path

def create_dynamic_yaml(img_path):
    """
    Tự động tạo file data.yaml dựa trên thư mục ảnh tìm thấy.
    """
    # Lấy thư mục cha của thư mục chứa ảnh để làm 'path' gốc
    # Ví dụ: nếu img_path là .../images/train thì root là .../images
    data_root = img_path.parent.parent if "train" in img_path.parts else img_path.parent
    
    # Đường dẫn tương đối từ vị trí script đến thư mục ảnh
    rel_train_path = os.path.relpath(img_path, current_dir := Path.cwd())
    
    yaml_content = {
        'path': str(current_dir), # Sử dụng thư mục gốc của script
        'train': rel_train_path,
        'val': rel_train_path,    # Dùng chung dữ liệu để test nhanh
        'names': {0: 'roblox player'}
    }

    with open('data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Đã cập nhật cấu hình động vào: data.yaml")

def main():
    img_path = setup_flexible_paths()
    if not img_path:
        return

    # Tạo file cấu hình phù hợp với máy hiện tại
    create_dynamic_yaml(img_path)

    # Chọn thiết bị (ưu tiên GPU)
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Đang chạy huấn luyện trên: {device}")

    # Khởi tạo YOLOv8
    model = YOLO('yolov8n.pt')

    try:
        model.train(
            data='data.yaml',
            epochs=300,             # Thay đổi số lượng epochs tùy nhu cầu
            imgsz=640,
            batch=16,
            device=device,
            project='roblox_runs',
            name='exp_unified',
            exist_ok=True,
            workers=0             # 0 là an toàn nhất cho cả Win và Linux
        )
        print("--- Huấn luyện hoàn tất thành công! ---")
    except Exception as e:
        print(f"Lỗi trong quá trình huấn luyện: {e}")

if __name__ == '__main__':
    main()
