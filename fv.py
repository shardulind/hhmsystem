import os
import cv2
import numpy as np
import faceRecognition as fr
from firebase import firebase
import datetime
import time

firebase = firebase.FirebaseApplication('https://hand-wash-record.firebaseio.com')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('/home/shardulind/Abhyas/cv/ei/trainingData.yml')

name = {0: "Ankit", 1:"Shardul"}


# To ensure that the person standing infront was actuall,, and to avoid error
ran_data = [0] * len(name.keys())
t0=0
isPersonNot = 0

cap=cv2.VideoCapture(0)
predicted_name=''
while True:

    ret,test_img = cap.read()
    faces_detected,gray_img=fr.faceDetection(test_img)

    for (x,y,w,h) in faces_detected:
        t0 = time.clock()
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
    resized_img=cv2.resize(test_img,(1000,700))
    cv2.imshow('face detection ', resized_img)
    cv2.waitKey(10)
    #print(time.clock() - t0)
    if faces_detected == () and (time.clock() - t0) > 2 and max(ran_data) != 0:
        result = firebase.post(name[ran_data.index(max(ran_data))], datetime.datetime.now())
        ran_data = [0]*len(name.keys())
        print("Data added",time.clock())


    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)
        print("confidence:",confidence)
        print("label:",label)
        print("Name:",name[label])

        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        if confidence < 100:
            fr.put_text(test_img,predicted_name,x,y)
            ran_data[label]+=1;
        #    result = firebase.post(predicted_name,datetime.datetime.now())
    resized_img=cv2.resize(test_img, (1000,700))
    cv2.imshow('face recognition ', resized_img)

    if cv2.waitKey(10) == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows
print(ran_data)
