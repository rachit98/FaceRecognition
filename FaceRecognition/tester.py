import cv2
import os
import numpy as np
import FaceRecognition as fr

test_img = cv2.imread('C:/Users/Acer/Desktop/FaceFace/FaceRecognition/FaceRecognition/rachit10.jpg')
faces_detected, gray_img = fr.faceDetection(test_img)

#for (x,y,w,h) in faces_detected:
#	cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=2)
#test_img = cv2.resize(test_img,(500,500))
#cv2.imshow("Rachit", test_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows

#faces,faceID = fr.labels_for_training_data('C:/Users/Acer/Desktop/FaceFace/FaceRecognition/FaceRecognition/Image')

#face_recognizer = fr.train_classifier(faces,faceID)

#face_recognizer.save('traindata.yml')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/Acer/Desktop/FaceFace/FaceRecognition/FaceRecognition/traindata.yml')
name = {1:"Andy",0:"Rachit"}

for fs in faces_detected:
    (x,y,w,h) = fs
    roi_gray = gray_img[y:y+h,x:x+h]
    label,confidence = face_recognizer.predict(roi_gray)
    fr.draw_rect(test_img,fs)
    predicted_name = name[label]
    #if confidence>37:
    #    continue
    fr.put_text(test_img,predicted_name,x,y)
    
test_img = cv2.resize(test_img,(500,500))
cv2.imshow("Images", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows
