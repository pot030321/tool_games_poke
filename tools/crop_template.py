import os
import time
from datetime import datetime
from pathlib import Path

ADB_PATH = r"C:\Users\PC\Downloads\platform-tools\adb.exe"  # chỉnh nếu khác

SAVE_DIR = Path("../data/raw")
INTERVAL = 3.0  # giây giữa mỗi lần chụp
LIMIT = None  # số ảnh tối đa (hoặc bỏ nếu muốn chạy vô hạn)


def ensure_dir(path):
    if not path.exists():
        path.mkdir(parents=True)
        print(f"📁 Tạo thư mục: {path}")


def capture_frame(index):
    temp_path = "screen.png"
    save_path = SAVE_DIR / f"frame_{index:05d}.png"
    os.system(f'"{ADB_PATH}" shell screencap -p /sdcard/screen.png')
    os.system(f'"{ADB_PATH}" pull /sdcard/screen.png {temp_path}')
    os.rename(temp_path, save_path)
    print(f"📸 Đã lưu: {save_path}")


def main():
    ensure_dir(SAVE_DIR)
    print(f"🚀 Bắt đầu chụp UI vào: {SAVE_DIR}")

    index = 1
    while True:
        capture_frame(index)
        index += 1
        if LIMIT and index > LIMIT:
            print("✅ Đã hoàn tất chụp.")
            break
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
