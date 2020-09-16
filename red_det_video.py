import cv2
import numpy as np
import os
import time
import timeit


def detect_red_circle():
    i = 0 ## counter for loop timer
    cam = cv2.VideoCapture("TEST_VIDEOS/DJI_0062.MOV")
    ret, img = cam.read()
    while ret:
        execute_start = time.time()
        img = cv2.resize(img, (640, 480))
        #img = cv2.imread("RtNrs.png")
        #img = cv2.GaussianBlur(img,(25,25), 0)
        #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #lower_range = np.array([0,150,150], dtype = np.uint8)
        #upper_range = np.array([180,255,255], dtype= np.uint8)

        #new_img = cv2.inRange(hsv, lower_range, upper_range)
        #new_img = cv2.medianBlur(new_img,5)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	    # Generating mask to detect red color
        lower_red = np.array([0,120,70])
        upper_red = np.array([10,255,255])
        mask1 = cv2.inRange(hsv,lower_red,upper_red)

        lower_red = np.array([170,120,70])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv,lower_red,upper_red)

        mask1 = mask1+mask2

	    # Refining the mask corresponding to the detected red color
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=2)
        mask1 = cv2.dilate(mask1,np.ones((3,3),np.uint8),iterations = 1)
        #mask1 = cv2.medianBlur(mask1,5)
        #mask1 = cv2.GaussianBlur(mask1,(5,5),cv2.BORDER_DEFAULT)

        ret, thresh = cv2.threshold(mask1, 50, 255, cv2.THRESH_BINARY)
        contours, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        area = []
        if contours:
            for contour in contours:
                area.append(cv2.contourArea(contour))
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(mask1, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        execute_end = time.time()
        print('Loop '+ str(i) + ' Execution Time:' +str((execute_end-execute_start))+' Seconds')
        print('Loop '+ str(i) + ' Contour Area'+ str(area))
        print()
        i+=1

        cv2.imshow('MASK', mask1)
        cv2.imshow('CIRCLES', img)

        if(cv2.waitKey(10) & 0xFF == ord('q')):
            break
        ret, img = cam.read()

detect_red_circle()
