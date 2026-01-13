import cv2
import mss
import numpy as np
import os
import time
import keyboard # Cài bằng: pip install keyboard

# Tạo thư mục lưu ảnh nếu chưa có
SAVE_PATH = "roblox_data"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

print("Đang chạy... Vào game Roblox và nhấn 'K' để chụp màn hình. Nhấn 'ESC' để thoát.")

with mss.mss() as sct:
    # Thiết lập vùng chụp (toàn màn hình hoặc tùy chỉnh)
    monitor = sct.monitors[1] 
    
    count = 0
    while True:
        if keyboard.is_pressed('k'):
            # Chụp màn hình
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) # Chuyển về BGR cho OpenCV
            
            # Lưu ảnh
            filename = f"{SAVE_PATH}/img_{int(time.time())}_{count}.jpg"
            cv2.imwrite(filename, img)
            
            print(f"Đã lưu: {filename}")
            count += 1
            time.sleep(0.3) # Đợi một chút để tránh trùng ảnh

        if keyboard.is_pressed('esc'):
            break

print(f"Hoàn tất! Bạn đã thu thập được {count} ảnh trong thư mục {SAVE_PATH}.")