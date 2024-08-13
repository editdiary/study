from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time

camera = Picamera2()

frame_resolution = camera.sensor_resolution
frame_rate = 16

# initialize the camera and grab a reference to the raw camera capture
camera_config = camera.create_preview_configuration(main={"size": frame_resolution, "format": "XRGB8888"}, controls={"FrameRate": frame_rate})
camera.configure(camera_config)
camera.start()

# allow the camera to warmup
time.sleep(1)

# capture frames from the camera
while True:
    # Capture the image
    image = camera.capture_array()

    # Convert the image from RGB to BGR (OpenCV uses BGR format)
    #image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Resize the image for display (optional)
    image = cv.resize(image, (600, 600))

    # hsv transform - value = gray image
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hue, saturation, value = cv.split(hsv)

    # kernel to use for morphological operations
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # applying topHat operations
    topHat = cv.morphologyEx(value, cv.MORPH_TOPHAT, kernel)

    # applying blackHat operations
    blackHat = cv.morphologyEx(value, cv.MORPH_BLACKHAT, kernel)

    # add and subtract between morphological operations
    add = cv.add(value, topHat)
    subtract = cv.subtract(add, blackHat)

    # applying gaussian blur on subtract image
    blur = cv.GaussianBlur(subtract, (5, 5), 0)

    # thresholding
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 19, 9)
    cv.imshow('thresh', thresh)

    key = cv.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv.destroyAllWindows()
camera.stop()