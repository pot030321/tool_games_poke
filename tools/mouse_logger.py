import os
import time
import json
import cv2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import threading
import keyboard
from pynput import mouse
from core.adb import adb_screenshot

SAVE_DIR = Path("../data/review")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = SAVE_DIR / "click_logs.json"

click_logs = []


def monitor_exit():
    print("❌ Nhấn phím 'q' để thoát...")
    keyboard.wait('q')
    print("🛑 Đã dừng.")
    os._exit(0)

def on_click(x, y, button, pressed):
    if pressed:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"click_{timestamp}.png"
        print(f"🖱️ Click tại ({x},{y}) → chụp ảnh: {filename}")

        img = adb_screenshot(save_path=SAVE_DIR / filename)
        if img is not None:
            click_logs.append({
                "image": filename,
                "action": f"tap:{x},{y}"
            })

        with open(LOG_PATH, "w") as f:
            json.dump(click_logs, f, indent=2)

if __name__ == "__main__":
    print("📍 Theo dõi click chuột → tự chụp màn & gán nhãn")
    print("⌛ Đang chạy... nhấn q để dừng")
    threading.Thread(target=monitor_exit, daemon=True).start()
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
