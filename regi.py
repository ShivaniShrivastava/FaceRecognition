#Register in DB
import cv2
import os
import numpy as np
from PIL import Image
import posix

face_cascade=cv2.CascadeClassifier('/home/shivani/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml')
#eye_cascade=cv2.CascadeClassifier('/home/shivani/opencv-3.2.0/data/haarcascades/haarcascade_eye.xml')
cap=cv2.VideoCapture(0)

id=input('Enter the user id : ')
sampleNum=0;
while True:
    ret,img=cap.read()
    #img=img[200:400,100:300]
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)
    gray=cv2.cvtColor(miniframe,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    for(x,y,w,h) in faces :
        sampleNum=sampleNum+1;
        #box = (100,100,400,400)
        #region = Image.resize((100,100))
        #img1=gray[x:x+400,y:y+400]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  #Blue color 255,0,0
        f=cv2.resize(gray[y:y+h,x:x+w],(200,200))
        cv2.imwrite("/home/shivani/Desktop/faces/User."+str(id)+"."+str(sampleNum)+".pgm",f)
        #img.thumbnail((14,14))
        #cv2.imshow("Face",img)
        # roi_gray=gray[y:y+h,x:x+w]
        # roi_color=img[y:y+h,x:x+w]
        # eyes=eye_cascade.detectMultiScale(roi_gray,scaleFactor=1.3,minNeighbors=5,minSize=(45,45),flags=cv2.CASCADE_SCALE_IMAGE)
        # for(ex,ey,ew,eh) in eyes:
        #      cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
       
    cv2.imshow('img',img)
    #k=cv2.waitKey(0)    #infinte waiting
    cv2.waitKey(100)#if i press q it'll break
    if(sampleNum>50):
        break   
    #if k == 27:
    #    break

cap.release()
cv2.destroyAllWindows()   

