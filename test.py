import os
from pathlib import Path
from ultralytics import YOLO
import cv2

def test_single_image():
    # 1. Đường dẫn đến ảnh cụ thể của bạn
    # Sử dụng Path để tự động xử lý dấu xuyệt trên Windows/Linux
    current_dir = Path.cwd()
    image_path = current_dir / "images" / "train" / "img_1768112676_32.jpg"
    
    # 2. Đường dẫn đến model vừa huấn luyện xong
    model_path = current_dir / "roblox_runs" / "exp_unified" / "weights" / "best.pt"

    if not model_path.exists():
        print(f"LỖI: Không tìm thấy file model tại {model_path}")
        print("Hãy chắc chắn bạn đã hoàn thành huấn luyện ít nhất 1 lần.")
        return

    if not image_path.exists():
        print(f"LỖI: Không tìm thấy ảnh tại {image_path}")
        return

    print(f"--- Đang kiểm tra ảnh: {image_path.name} ---")

    # 3. Tải model
    model = YOLO(model_path)

    # 4. Chạy nhận diện
    # conf=0.25: Chỉ hiện những vật thể AI tự tin trên 25%
    # ... (giữ nguyên phần trên)
    # Thử hạ thấp conf xuống 0.05 để xem AI có đoán được gì không
    results = model.predict(source=str(image_path), conf=0.02, save=True, project="test_results", name="single_test")


    # 5. Thông báo kết quả
    for result in results:
        # Đường dẫn thư mục lưu kết quả (YOLO tự tạo)
        save_dir = Path(result.save_dir)
        print(f"\nThành công!")
        print(f"Số lượng nhân vật tìm thấy: {len(result.boxes)}")
        print(f"Ảnh kết quả đã được lưu tại: {save_dir / image_path.name}")

        # Hiển thị ảnh nếu đang chạy trên máy tính có màn hình (Windows)
        if os.name == 'nt':
            img_res = cv2.imread(str(save_dir / image_path.name))
            cv2.imshow("Ket qua kiem tra AI", img_res)
            print("Nhấn phím bất kỳ trên cửa sổ ảnh để đóng.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    test_single_image()