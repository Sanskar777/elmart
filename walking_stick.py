import cv2 as cv
import numpy as np
# from matlplotlib import pyplot as plt

image = cv.imread('room1.jpg')
cv. imshow("window", image)
image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
cv. imshow("window2", image)
# _,threshold = cv.adaptiveThreshold(image,127,255,0)
# ret2,threshold = cv.threshold(image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
blur = cv.GaussianBlur(image,(5,5),0)
ret3,threshold = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# cv.adaptiveThreshold(img,255,cv.ADAP/TIVE_THRESH_GAUSSIAN_C,\
            # cv.THRESH_BINARY,11,2)
image2, contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print contours
print 'length of contours is ', len(contours)
for i in range(len(contours)):
    # col = np.random.randint(0,256,size=3)
    cv.drawContours(image, contours, i, (0,255,0), 5)
    print cv.contourArea(contours[i])
    rect = cv.boundingRect(contours[i])
    x,y,h,w = rect
    if ( rect[2]>20 and rect[3]>20):
        cv.rectangle(image,(x,y),(x+h,y+w),(25,5,100),3)
cv.imshow('contours',image)
cv.imshow("show",image)
# cv. imshow("window3", image)
cv.imshow("window3", image2)
cv.waitKey(0)
cv.destroyAllWindows()
