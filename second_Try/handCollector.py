import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import time


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
 
offset = 20
imgSize = 300

DATA_DIR = 'D:\dataset1'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


number_of_classes = 5
dataset_size = 500

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('Capture Dataset', frame)
        if cv2.waitKey(25) == ord('q'):
            break
    counter = 0
    while counter < dataset_size:
        success,img = cap.read()
        hands,img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x,y,w,h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            
            imgCropShape = imgCrop.shape
            
            aspectRatio = h/w
            # height 
            
            if aspectRatio >1:
                k = imgSize /h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop,(wCal,imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
            
            # width
            if aspectRatio <1:
                k = imgSize /w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop,(imgSize,hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap:] = imgResize
            
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite) 
        
            
        cv2.imshow("Image", img)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)),imgWhite)
        print(counter)
        counter += 1