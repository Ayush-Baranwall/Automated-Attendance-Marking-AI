import numpy as np
import cv2
import os
# ↑ above is all import files ↑

# getting file dir cuz path is diffecrent in our PCs
file_directory = os.getcwd().replace('\\', '/')
#using opencv frontal face template to recognize face (note: only detects front face)
face_cascade = cv2.CascadeClassifier(f'{file_directory}/cascades/data/haarcascade_frontalface_alt2.xml')
# ↓ this takes pics ↓
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read() # taking images frame by frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converts image to black and white 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) # uses opencv template detects frontal face then puts it in faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        cv2.imwrite('ai_ml_project/my_img.png', roi_gray) # saves detected faces

        color = (100, 255, 100) # this is in BGR 
        stroke = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)

    cv2.imshow('frame', frame) # this opens camera and show us
    if cv2.waitKey(20) & 0xFF == ord('q'):  # no idea why we use this but required to run 
        break

cap.release()   # like free() in C
cv2.destroyAllWindows() # closes all windows 