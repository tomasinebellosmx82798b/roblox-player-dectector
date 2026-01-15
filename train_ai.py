import os
import torch
import yaml
import zipfile
from pathlib import Path
from ultralytics import YOLO

def extract_datasets():
    """Giải nén datasets.zip và chuẩn hóa nhãn."""
    zip_path = Path("datasets.zip")
    extract_to = Path("datasets_extracted")
    
    if not zip_path.exists():
        print("LỖI: Không tìm thấy file datasets.zip!")
        return None

    print(f"--- Đang giải nén {zip_path} ---")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Tự động tìm thư mục images và labels bên trong
    # Cấu trúc mong muốn: datasets_extracted/images/train và datasets_extracted/labels/train
    img_train_path = extract_to / "images" / "train"
    lbl_train_path = extract_to / "labels" / "train"

    if not img_train_path.exists() or not lbl_train_path.exists():
        print("CẢNH BÁO: Cấu trúc file trong zip không đúng chuẩn (images/train, labels/train).")
        # Thử tìm kiếm sâu hơn nếu user nén lệch thư mục
        all_imgs = list(extract_to.rglob("*.jpg")) + list(extract_to.rglob("*.png"))
        if all_imgs:
            img_train_path = all_imgs[0].parent
            lbl_train_path = img_train_path.parent.parent / "labels" / "train"
            print(f"Đã tìm thấy ảnh tại: {img_train_path}")

    # --- TỰ ĐỘNG SỬA ID NHÃN VỀ 0 ---
    print("--- Đang chuẩn hóa ID nhãn về 0 ---")
    if lbl_train_path.exists():
        for lbl_file in lbl_train_path.glob("*.txt"):
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = "0" # Ép ID về 0
                    new_lines.append(" ".join(parts))
            with open(lbl_file, 'w') as f:
                f.write("\n".join(new_lines) + "\n")

    return extract_to

def create_yaml(dataset_root):
    """Tạo file data.yaml trỏ vào thư mục đã giải nén."""
    current_dir = Path.cwd()
    # Đường dẫn tương đối từ vị trí chạy script
    yaml_content = {
        'path': str(current_dir / dataset_root),
        'train': 'images/train',
        'val': 'images/train', # Dùng tạm tập train làm tập val để tiết kiệm ảnh
        'names': {0: 'roblox player'}
    }
    with open('data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    return 'data.yaml'

def main():
    # 1. Giải nén dữ liệu
    dataset_root = extract_datasets()
    if not dataset_root:
        return

    # 2. Tạo cấu hình
    create_yaml(dataset_root)

    # 3. Thiết bị huấn luyện
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    # 4. LOAD MODEL: Ưu tiên file cùng thư mục
    # Không cần tính năng thông minh, cứ lấy trực tiếp file trong folder
    if os.path.exists('best.pt'):
        model_path = 'best.pt'
        print(f"--- Đang HUẤN LUYỆN TIẾP TỤC từ: {model_path} ---")
    else:
        model_path = 'yolov8n.pt'
        print(f"--- Đang HUẤN LUYỆN MỚI từ: {model_path} ---")
    
    model = YOLO(model_path) 

    # 5. Chạy huấn luyện
    model.train(
        data='data.yaml',
        epochs=300,             # Huấn luyện thêm 50 lần
        imgsz=640,
        batch=16,
        device=device,
        project='roblox_runs',
        name='exp_finetuned',
        exist_ok=True,
        # Các tham số tăng cường để nhận diện đa dạng hơn
        hsv_h=0.015, 
        hsv_s=0.7,   
        hsv_v=0.4,   
        degrees=10.0,
        fliplr=0.5,  
        mosaic=1.0
    )
    print("--- Hoàn tất! Check kết quả trong thư mục roblox_runs/exp_finetuned ---")

if __name__ == '__main__':
    main()
