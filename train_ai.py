from ultralytics import YOLO
import torch
import os

def main():
    # 1. Kiểm tra thiết bị phần cứng (Ưu tiên GPU NVIDIA để train nhanh)
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Đang sử dụng thiết bị: {device}")

    # 2. Chọn mô hình nguồn
    # Nếu bạn đã có file 'best.pt' từ lần trước và muốn train tiếp, 
    # hãy đổi 'yolov8n.pt' thành đường dẫn tới file đó (ví dụ: 'roblox_esp/run_1/weights/best.pt')
    model_path = 'yolov8n.pt' 
    
    # Kiểm tra xem có file cũ để train tiếp không (tùy chọn)
    # model_path = 'roblox_esp/run_1/weights/best.pt' if os.path.exists('roblox_esp/run_1/weights/best.pt') else 'yolov8n.pt'

    model = YOLO(model_path)

    # 3. Bắt đầu huấn luyện
    # - data: file cấu hình data.yaml
    # - epochs: số lần học lại toàn bộ dữ liệu (tăng lên nếu AI vẫn nhận diện kém)
    # - imgsz: kích thước ảnh đầu vào (640 là chuẩn cho YOLOv8)
    # - workers: số luồng xử lý dữ liệu (nên để 2-4 tùy cấu hình máy)
    results = model.train(
        data='data.yaml', 
        epochs=50, 
        imgsz=640, 
        device=device,
        project='roblox_esp',
        name='run_1',
        exist_ok=True # Ghi đè lên thư mục run_1 nếu đã tồn tại thay vì tạo run_2
    )

    print("-" * 30)
    print("Huấn luyện hoàn tất!")
    print(f"File AI của bạn (best.pt) đã được xuất ra tại: roblox_esp/run_1/weights/best.pt")
    print("Bạn hãy giữ file 'best.pt' này để chúng ta lập trình ứng dụng ESP ở bước tiếp theo.")

if __name__ == '__main__':
    main()