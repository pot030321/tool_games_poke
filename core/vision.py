import cv2
import numpy as np
from pathlib import Path


def match_template(screen, template_path, threshold=0.8):
    template = cv2.imread(str(template_path), 0)
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    h, w = template.shape
    if max_val >= threshold:
        return True, max_loc, (w, h)
    else:
        return False, None, (w, h)


def click_template(screen, template_path, adb_tap_func, threshold=0.8):
    found, pos, (w, h) = match_template(screen, template_path, threshold)
    if found:
        center_x = pos[0] + w // 2
        center_y = pos[1] + h // 2
        adb_tap_func(center_x, center_y)
        return True
    return False


def crop_template(screen, x, y, w, h, save_path):
    crop = screen[y:y + h, x:x + w]
    cv2.imwrite(str(save_path), crop)
    return save_path
