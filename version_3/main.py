import cv2
import pandas as pd
import glob
import os
import cv2

from deepface import DeepFace
from retinaface import RetinaFace

image_path = "E:\\my_projects\\ai_ml_project\\version_3\\test.jpeg"
faces = RetinaFace.detect_faces(image_path)
image = cv2.imread(image_path)

count = 0
for key in faces.keys():
    face = faces[key]
    facial_area = face["facial_area"]
    crop_image = image[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]
    cv2.imwrite(f'croped_images\\image_{count}.png', crop_image)
    count += 1

data_collected = glob.glob("E:\\my_projects\\ai_ml_project\\version_3\\croped_images\\*")
data_set = glob.glob("E:\\my_projects\\ai_ml_project\\version_3\\data_set\\*")

names = list()
for image_data in data_set:
    names.append(image_data.split('\\')[-1].split('.')[0])
    
df = pd.DataFrame({'name': names})
df['attendance'] = 0

for image_collected in data_collected:
    for image_data in data_set:
        isSame = DeepFace.verify(img1_path=image_collected, img2_path=image_data, enforce_detection=False)['verified']
        isSame_twice = DeepFace.verify(img1_path=image_collected, img2_path=image_data, model_name='ArcFace', enforce_detection=False)['verified']
        if isSame and isSame_twice:
            print(image_collected, image_data)
            name = image_data.split('\\')[-1].split('.')[0]
            df['attendance'][df.loc[df['name'] == name].index[0]] = 1
            
df.to_excel("attendance.xlsx", index=False)