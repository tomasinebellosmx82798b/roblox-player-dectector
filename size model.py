import os
import sys
import site
from pathlib import Path

def get_size(path):
    """Tính toán dung lượng thư mục theo bytes."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    except (PermissionError, FileNotFoundError):
        return 0
    return total_size

def main():
    # Lấy danh sách các thư mục chứa thư viện (site-packages)
    paths = site.getusersitepackages()
    if isinstance(paths, str):
        paths = [paths]
    
    # Nếu không tìm thấy user site, thử lấy system site
    if not paths:
        paths = site.getsitepackages()

    print(f"Đang quét các thư mục thư viện...")
    
    package_list = []
    
    for base_path in paths:
        base_path = Path(base_path)
        if not base_path.exists():
            continue
            
        print(f"Đang kiểm tra: {base_path}")
        
        # Duyệt qua các thư mục con trong site-packages
        for item in base_path.iterdir():
            if item.is_dir():
                # Bỏ qua các thư mục dist-info hoặc __pycache__
                if item.name.endswith('.dist-info') or item.name == '__pycache__':
                    continue
                
                size_mb = get_size(item) / (1024 * 1024)
                if size_mb > 0.1:  # Chỉ lấy các thư viện > 0.1 MB
                    package_list.append((item.name, size_mb))

    # Sắp xếp theo dung lượng giảm dần
    package_list.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*50)
    print(f"{'Thư viện (Thư mục)':<30} | {'Dung lượng (MB)':<15}")
    print("-" * 50)

    total_all = 0
    for name, size in package_list:
        print(f"{name:<30} | {size:>10.2f} MB")
        total_all += size

    print("-" * 50)
    print(f"{'TỔNG CỘNG':<30} | {total_all:>10.2f} MB")
    print("="*50)
    print("\nGợi ý: Nếu 'torch' quá nặng (>2000MB) và bạn không dùng GPU, hãy cài bản CPU-only.")
    print("Xóa thư viện thừa bằng lệnh: pip uninstall <tên_thư_viện>")

if __name__ == "__main__":
    main()