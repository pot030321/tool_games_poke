import os
import cv2

ADB_PATH = r"C:\Users\PC\Downloads\platform-tools\adb.exe"


def adb_screenshot(save_path="screen_now.png"):
    os.system(f'"{ADB_PATH}" shell screencap -p /sdcard/screen.png')
    os.system(f'"{ADB_PATH}" pull /sdcard/screen.png {save_path}')
    return cv2.imread(save_path)


def adb_tap(x, y):
    os.system(f'"{ADB_PATH}" shell input tap {x} {y}')


def adb_start_game(package_name="com.pokedaigpguanwang.vn"):
    os.system(f'"{ADB_PATH}" shell monkey -p {package_name} -c android.intent.category.LAUNCHER 1')
