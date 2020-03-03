import cv2
import os
import numpy as np
import FaceRecognition as fr

test_img = cv2.imread('C:/Users/rachit.yagnik/Desktop/FaceRecognition/Image/rachit1.jpg')
faces_detected, gray_img = fr.faceDetect(test_img)

for (x,y,w,h) in faces_detected:
	cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=2)
test_img = cv2.resize(test_img,(500,500))
cv2.imshow("Rachit", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows