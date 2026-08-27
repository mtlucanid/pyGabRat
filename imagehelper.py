#
# imagehelper.py
#
#  Helper module for image window output
#
#  Copyright (C) 2026, Masahiko TANAHASHI
#
import cv2
from screeninfo import get_monitors

###########################################
## Constants
IMSHOW_MIN_WINDOW_SIZE = 160

###########################################
## Helper functions for image outputs
def imshow_init(img):
    """Prepare a fixed-sized window to show animations"""
    # Close previous windows
    cv2.destroyAllWindows()
    # Fetch the primary monitor's resolution
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height
    height, width = img.shape[:2]
    # Resize the image so that it does not exceed the screen size
    if width > screen_width:
        scale = screen_width / width
        width *= scale
        height *= scale
    if height > screen_height:
        scale = screen_height / height
        width *= scale
        height *= scale
    # Resize the image so that it is not smaller than the minimum size
    if width < IMSHOW_MIN_WINDOW_SIZE:
        scale = IMSHOW_MIN_WINDOW_SIZE / width
        width *= scale
        height *= scale
    if height < IMSHOW_MIN_WINDOW_SIZE:
        scale = IMSHOW_MIN_WINDOW_SIZE / height
        width *= scale
        height *= scale
    # Precreate the named window with a fixed size (to avoid window flashing)
    width = int(width)
    height = int(height)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", width, height)

def imshow(img, wait):
    cv2.imshow("Image", img)
    if wait > 0:
        cv2.waitKey(wait)

def imshow_wait(wait):
    if wait > 0:
        cv2.waitKey(wait)

def imshow_key():
    cv2.waitKey(0)

def imshow_exit():
    cv2.destroyAllWindows()
