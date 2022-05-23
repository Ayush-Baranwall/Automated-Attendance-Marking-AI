import time
import cv2
import pandas as pd
import glob
import os
import cv2

file_directory = os.path.dirname(os.path.abspath(__file__)) # gets us file's path 

from deepface import DeepFace  # for recognition
from retinaface import RetinaFace # for detection
# deepface and retinaface are huge files 
# so first run might take 3 min then it will take 10 sec 

print("imported files")

camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read() # taking images frame by frame
    cv2.imwrite('image.png', frame)
    cv2.imshow("Capturing", frame)  # this opens camera and show us
    # time.sleep(10)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
del(camera)

image_path = f"{file_directory}\\image.png"  # input image 
faces = RetinaFace.detect_faces(image_path)  
image = cv2.imread(image_path)

print("detecting and reading faces")

count = 0  
for key in faces.keys():
    face = faces[key]  # getting each face from the image
    facial_area = face["facial_area"] # marking the area where face lie
    crop_image = image[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]] # croping those faces 
    cv2.imwrite(f'croped_images\\image_{count}.png', crop_image) # saving them to recognise 
    count += 1

print("croped faces are stored")

data_collected = glob.glob(f"{file_directory}\\croped_images\\*")
data_set = glob.glob(f"{file_directory}\\data_set\\*")

names = list()   # names list to mark attendance 
for image_data in data_set:
    names.append(image_data.split('\\')[-1].split('.')[0]) # used to get name of the file 
    
df = pd.DataFrame({'name': names}) # our temp attendance sheet
df['attendance'] = 0 # initially all are absent 
print("starting to recognise faces")

for image_collected in data_collected:
    for image_data in data_set:
        # comparing all croped faces with our dataset(database) 
        isSame = DeepFace.verify(img1_path=image_collected, img2_path=image_data, enforce_detection=False)['verified'] # default is VGG- face
        isSame2 = DeepFace.verify(img1_path=image_collected, img2_path=image_data, model_name='ArcFace', enforce_detection=False)['verified']
        if isSame and isSame2:
            name = image_data.split('\\')[-1].split('.')[0]
            print(f'found {name}')
            df['attendance'][df.loc[df['name'] == name].index[0]] = 1 
            # marking attendance
            
print("recognized faces and attendance marked")
df.to_excel("attendance.xlsx", index=False) # saving them attendance 

remove_dir = glob.glob(f"{file_directory}\\croped_images\\*")
for f in remove_dir:
    os.remove(f)
    # remove croped faces and make room for future croped faces

print("croped images used to mark attendance is removed")

# done