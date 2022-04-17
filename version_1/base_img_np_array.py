import os
import cv2
import numpy as np
import pickle # pip install pillow
from PIL import Image   

# ↑ import files ↑

file_directory = os.path.dirname(os.path.abspath(__file__)) # getting location of faces folder
faces_to_train_directory = os.path.join(file_directory, 'faces_to_train')
face_cascade = cv2.CascadeClassifier(f'{file_directory}\\cascades\\data\\haarcascade_frontalface_alt2.xml')

id = 0 # used to give id for each face in faces_to_train directory
label_ids = dict()
y_labels = list()
x_train = list()

for root, dirs, files in os.walk(faces_to_train_directory):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            # print(label, path)
            if not label in label_ids:
                label_ids[label] = id
                id += 1 

            file_face_id = label_ids[label]

            pil_image = Image.open(fr'{path}').convert('L')  # .conver('L') converts it into black and white
            image_array = np.array(pil_image, 'uint8')  # converting images to numpy array 
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w] # roi => region of interest 
                x_train.append(roi)
                y_labels.append(file_face_id)

with open('labels.pickel', 'wb') as f:
    pickle.dump(label_ids, f)
