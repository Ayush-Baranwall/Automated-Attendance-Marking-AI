import numpy as np
import cv2
import os
import time # to run file for specific time 
import pickle
# ↑ above is all import files ↑
name = str()
# using opencv recognizer for this run
recognizer = cv2.face.LBPHFaceRecognizer_create()  # pip install opencv-contrib-python | to make it work
recognizer.read('LBPH_train.yml')
labels = dict()
with open('labels.pickel', 'rb') as f:  # reading from pickle 
    file_lables = pickle.load(f)
    # reversing the file lables cuz its name : id, we want id : name cuz we have id from yml file but we need to show name
    labels = {v:k for k, v in file_lables.items()} 


# getting file dir cuz path is diffecrent in our PCs
file_directory = os.path.dirname(os.path.abspath(__file__))
#using opencv frontal face template to recognize face (note: only detects front face)
face_cascade = cv2.CascadeClassifier(f'{file_directory}\\cascades\\data\\haarcascade_frontalface_alt2.xml')
# ↓ this takes pics ↓
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # cv2.VideoCapture(0)

t_end = time.time() + 50
while True:
    ret, frame = cap.read() # taking images frame by frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converts image to black and white 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) # uses opencv template detects frontal face then puts it in faces
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]  # ROI stands for region of interest 
        cv2.imwrite('image.png', roi_color)

        # using opencv train instead of scikit learn here ↓
        id, confidence = recognizer.predict(roi_gray) # confidence is the predicted label !!! still need to figure out how it works 
        if  confidence > 40: # and confidence <= 85: 
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id]
            color = (0, 255, 0)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        # cv2.imwrite('ai_ml_project/my_img.png', roi_gray) # saves detected faces

        color = (100, 255, 100) # this is in BGR 
        stroke = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke) # this draws a rectangle from x, y to x+w, y+h with color and stroke

    cv2.imshow('frame', frame) # this opens camera and show us

    if time.time() > t_end:
        break

    if cv2.waitKey(20) & 0xFF == ord('q'):  # no idea why we use this but required to run 
        break

# face_file = f'{file_directory}\\{name}'
# shutil.copyfile(f'{file_directory}\\image.png', face_file)

cap.release()   # like free() in C
cv2.destroyAllWindows() # closes all windows 
