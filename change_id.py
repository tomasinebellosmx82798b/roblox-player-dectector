import os
from pathlib import Path

def fix_label_ids():
    """
    Quét tất cả file .txt trong thư mục labels và ép ID về 0.
    """
    # Tự động tìm thư mục labels dựa trên cấu trúc bạn đã mô tả
    current_dir = Path.cwd()
    
    # Danh sách các đường dẫn có thể chứa labels
    possible_paths = [
        current_dir / "roblox_esp_ai_legit" / "labels" / "train",
        current_dir / "roblox_esp_ai_legit" / "labels" / "roblox_data",
        current_dir / "labels" / "train",
        current_dir / "labels"
    ]

    label_dir = None
    for p in possible_paths:
        if p.exists() and p.is_dir():
            label_dir = p
            break

    if not label_dir:
        print("LỖI: Không tìm thấy thư mục nhãn (labels)!")
        return

    print(f"--- Bắt đầu sửa ID tại: {label_dir} ---")
    
    files = list(label_dir.glob("*.txt"))
    fixed_count = 0

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # ÉP ID VỀ 0: 
                    # Dù id cũ là 15 hay bất cứ số nào, nó sẽ thành 0
                    parts[0] = "0" 
                    new_lines.append(" ".join(parts))
            
            # Ghi đè lại file với ID đã sửa
            if new_lines:
                with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write("\n".join(new_lines) + "\n")
                fixed_count += 1
                
        except Exception as e:
            print(f"Lỗi khi xử lý {file_path.name}: {e}")

    print(f"Hoàn tất! Đã sửa ID thành 0 cho {fixed_count} file nhãn.")
    print("Gợi ý: Bây giờ bạn hãy xóa file .cache và chạy lại train_ai.py.")

if __name__ == "__main__":
    fix_label_ids()