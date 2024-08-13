from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import tensorflow as tf
import time

camera = Picamera2()
model = tf.keras.models.load_model('digits_model.h5')
SZ = 28

frame_resolution = camera.sensor_resolution
frame_rate = 16
margin = 30

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

    # Resize the image for display (optional)
    display_width, display_height = 600, 600
    image = cv.resize(image, (display_width, display_height))

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
    #cv.imshow('thresh', thresh)

    cv2MajorVersion = cv.__version__.split(".")[0]
    # check for contours on thresh
    if int(cv2MajorVersion) >= 4:
        contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    else:
        imageContours, contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    img_digits = []
    positions = []

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)

        # Ignore small sections
        if w * h < 2400: continue
        y_position = max(y-margin, 0)
        x_position = max(x-margin, 0)
        img_roi = thresh[y_position:y+h+margin, x_position:x+w+margin]
        num = cv.resize(img_roi, (SZ, SZ))      ## 학습 모델의 이미지 input이 28x28이므로
        num = num.astype('float32') / 255.

        result = model.predict(np.array([num]))
        result_number = np.argmax(result)
        cv.rectangle(image, (x-margin, y-margin), (x+w+margin, y+h+margin), (0, 255, 0), 2)

        text = f"Number is : {result_number}"
        cv.putText(image, text, (margin, display_height-margin), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # show the frame
    cv.imshow("MNIST Hand Write", image)
    key = cv.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv.destroyAllWindows()
camera.stop()