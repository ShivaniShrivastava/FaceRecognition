#Detector
import posixpath
import os
import cv2
import numpy as np
from PIL import Image

#recognizer=cv2.face.createEigenFaceRecognizer();
path="/home/shivani/Desktop/faces"

def getImagesWithID(path):
    imagePaths=[posixpath.join(path,f) for f in os.listdir(path)]
    facesh=[]
    IDs=[]
    for imagePath in imagePaths:
        #print(imagePath)
        faceImg=Image.open(imagePath).convert('L')
        #faceNp = faceImg[200:400, 100:300]
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        #ID=str(os.path.split(imagePath)[-1].split('.')[1])
        facesh.append(faceNp)
        #print(ID)
        IDs.append(ID)
        #cv2.imshow("training",faceNp)
        #cv2.waitKey(10)
    return IDs,facesh

# Ids,facesh=getImagesWithID(path)
# recognizer.train(facesh,np.array(Ids))
# recognizer.save('/home/shivani/opencv-3.2.0/recognizer/trainingData.yml')
# cv2.destroyAllWindows()
#Ids,facesh=getImagesWithID(path)

  #Printing the paths
# getImagesWithID(path)
def facerec():
    names=['Shivi','Unknown']
    Ids,facesh=getImagesWithID(path)
    recognizer = cv2.face.createEigenFaceRecognizer()
    recognizer.train(facesh,np.array(Ids))
    #recognizer.save('/home/shivani/opencv-3.2.0/recognizer/trainingData.yml')
#recognizer.load('trainer/trainingData.yml')
    cascadePath = "/home/shivani/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    cam = cv2.VideoCapture(0)
#font = cv2.InitFont(cv2.FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
    while True:
        ret,im =cam.read()
        minisize = (im.shape[1],im.shape[0])
        miniframe = cv2.resize(im, minisize)
        gray=cv2.cvtColor(miniframe,cv2.COLOR_BGR2GRAY)
        #gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            f=cv2.resize(gray[y:y+h,x:x+w],(200,200))
            Id, conf = recognizer.predict(f)
            print(conf)
            print(Id)
            # if(Id==1):
            #     Id="Shivi"
            #     print(Id)
            # elif(Id==2):
            #     Id="Manali"
            cv2.putText(im,str(Id), (x,y+h),cv2.FONT_HERSHEY_SIMPLEX,1, 255,2)
        cv2.imshow('im',im)
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

facerec()
#     X,Y=getImagesWithID(path)
#     recognizer=cv2.face.createEigenFaceRecognizer()
#     recognizer.train(Y,np.array(X))
#     camera=cv2.VideoCapture(0)
#     #print("Hey")
#     face_cascade=cv2.CascadeClassifier('/home/shivani/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml')
#     while True:
#         rd,img=camera.read()
#         faces=face_cascade.detectMultiScale(img,1.3,5)
#         for (x, y, w, h) in faces:
#             img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#             gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#             roi=cv2.resize(gray[y:y+h,x:x+w])
#             print("Hey0")
#             Id,conf=recognizer.predict(roi)
#             print("Hey")
#             print("Label =: %s, Confidence: %.2f" %Id %conf) 
#             print("Hey2")
#             cv2.cv.putText(img,names[params[0]],(x,y-20),cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
#             cv2.imshow("camera",img)
#             if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
#                 break
#         cv2.destroyAllWindows() 

# def facerec():


# def main():
#     facerec()

# main()   
# print("Hey")
# recognizer.train(facesh,np.array(Ids))
# recognizer.save('/home/shivani/opencv-3.2.0/recognizer/trainingData.yml')
# cv2.destroyAllWindows()



  #-Calling the function   
    
