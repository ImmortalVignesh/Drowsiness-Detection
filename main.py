from scipy.spatial import distance as dist #dist of coor
from imutils import  face_utils #coorder of different parts
import imutils
import dlib #face detection and landmarks on face
import cv2

import winsound

frequency=2500
duration=1000

def eyeAspectRation(eye):
    #vertical
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])
    #horizontal
    C=dist.euclidean(eye[0],eye[3])
    ear = (A+B)/(2.0*C)
    return ear

count = 0
earThresh = 0.3
earFrames=100
shapePredictor="shape_predictor_68_face_landmarks.dat"


cam=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(shapePredictor)

#get the coord of left & right eye

(lSatrt,lEnd)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rSatrt,rEnd)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    _,frame=cam.read()
    frame=imutils.resize(frame,width=450)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects=detector(gray,0)
    for rect in rects:
        shape=predictor(gray,rect)
        shape=face_utils.shape_to_np(shape)

        leftEye=shape[lSatrt:lEnd]
        rightEye=shape[rSatrt:rEnd]
        leftEAR= eyeAspectRation(leftEye)
        rightEAR= eyeAspectRation(rightEye)

        ear=(leftEAR+rightEAR)/2.0

        leftEyeHull=cv2.convexHull(leftEye)
        rightEyeHull=cv2.convexHull(rightEye)
        cv2.drawContours(frame,[leftEyeHull],-1,(0,0,255),1)
        cv2.drawContours(frame,[rightEyeHull],-1,(0,0,255),1)
        print(ear,earThresh)

        if ear<=earThresh:
            cv2.putText(frame,"DROWSINESS DETECTED",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            winsound.Beep(frequency,duration)
            print(ear,earThresh)
        else:
            count=0
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1)
    if key==ord("q"):
        break
cam.release()
cv2.destroyWindow()



