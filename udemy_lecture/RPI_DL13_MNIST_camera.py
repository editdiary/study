import cv2 as cv
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import tensorflow as tf
import time

model = tf.keras.models.load_model('digits_model.h5')
default_src = 0
title_name = 'MNIST Hand Write Digits Recognition Video'
SZ = 28
frame_width = 300
frame_height = 300
cap = cv.VideoCapture(default_src)

def start():
    # initialize the video stream and allow the camera sensor to
    # warmup
    global cap
    default_src = int(srcSpin.get())
    #cap = VideoStream(usePiCamera=1).start()
    cap = cv.VideoCapture(default_src)
    #time.sleep(2.0)
    detectAndDisplay()

def detectAndDisplay():
    _, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]
    width_ratio = frame_width / width
    height_ratio = frame_height / height

    frame = cv.resize(frame, (frame_width, frame_height))
    # hsv transform - value = gray image
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
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
    thresh = cv.adaptiveThreshold(blur, 255.0, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 19, 9)
    #cv.imshow('thresh', thresh)

    #cv.findCountours() function changed from OpenCV3 to OpenCV4: now it have only two parameters instead of 3
    cv2MajorVersion = cv.__version__.split(".")[0]
    # check for contours on thresh
    if int(cv2MajorVersion) >= 4:
        contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    else:
        imageContours, contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    img_digits = []
    positions = []
    margin = 30

    img_origin = frame.copy()

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)

        # Ignore small sections
        if w * h < 2400: continue
        y_position = y - margin
        if (y_position < 0): y_position = 0
        x_position = x - margin
        if (x_position < 0): x_position = 0
        img_roi = thresh[y_position:y+h+margin, x_position:x+w+margin]
        num = cv.resize(img_roi, (SZ, SZ))
        num = num.astype('float32') / 255.
        
        result = model.predict(np.array([num]))
        result_number = np.argmax(result)
        cv.rectangle(frame, (x-margin, y-margin), (x+w+margin, y+h+margin), (0, 255, 0), 2)
        y = y - 10 if y - 10 > 10 else y + 10
        text = f'Number is : {result_number}'
        cv.putText(frame, text, (margin, frame_height-margin), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detectAndDisplay)

#main
main = Tk()
main.title(title_name)
main.geometry()

#Graphics window
label = Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0, column=0, columnspan=4)
srcLabel = Label(main, text='Video Source : ')
srcLabel.grid(row=1, column=0)
srcVal = IntVar(value=default_src)
srcSpin = Spinbox(main, textvariable=srcVal, from_=0, to=5, increment=1, justify=RIGHT)
srcSpin.grid(row=1, column=1)

Button(main, text="Start", height=2, command=lambda:start()).grid(row=1, column=2, columnspan=2, sticky=(W, E))
imageFrame = Frame(main)
imageFrame.grid(row=2, column=0, columnspan=4)

#Capture video frames
lmain = Label(imageFrame)
lmain.grid(row=0, column=0)

main.mainloop()