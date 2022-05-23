import cv2
import pandas as pd
import glob
import os
import cv2

file_directory = os.path.dirname(os.path.abspath(__file__))

from deepface import DeepFace
from retinaface import RetinaFace

print("imported files")

image_path = f"{file_directory}\\test.jpeg"
faces = RetinaFace.detect_faces(image_path)
image = cv2.imread(image_path)

print("detecting and reading faces")

count = 0
for key in faces.keys():
    face = faces[key]
    facial_area = face["facial_area"]
    crop_image = image[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]
    cv2.imwrite(f'croped_images\\image_{count}.png', crop_image)
    count += 1

print("croped faces are stored")

data_collected = glob.glob(f"{file_directory}\\croped_images\\*")
data_set = glob.glob(f"{file_directory}\\data_set\\*")

names = list()
for image_data in data_set:
    names.append(image_data.split('\\')[-1].split('.')[0])
    
df = pd.DataFrame({'name': names})
df['attendance'] = 0
print("starting to recognise faces")
for image_collected in data_collected:
    for image_data in data_set:
        isSame = DeepFace.verify(img1_path=image_collected, img2_path=image_data, enforce_detection=False)['verified']
        isSame2 = DeepFace.verify(img1_path=image_collected, img2_path=image_data, model_name='ArcFace', enforce_detection=False)['verified']
        if isSame and isSame2:
            name = image_data.split('\\')[-1].split('.')[0]
            print(f'found {name}')
            df['attendance'][df.loc[df['name'] == name].index[0]] = 1
            
print("recognized faces and attendance marked")
df.to_excel("attendance.xlsx", index=False)

remove_dir = glob.glob(f"{file_directory}\\croped_images\\*")
for f in remove_dir:
    os.remove(f)

print("croped images used to mark attendance is removed")