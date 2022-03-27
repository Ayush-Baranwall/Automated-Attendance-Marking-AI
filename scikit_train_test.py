import os
import cv2
from nbformat import read
import numpy as np
import pickle 
from PIL import Image  
import csv 
import random
import json
import pandas as pd


from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

model = KNeighborsClassifier(n_neighbors=1)
# model = svm.SVC()
# model = Perceptron()


# big explaination
data = pd.read_csv('scikit_data.csv')
data['image'] = data['image'].transform(lambda x: np.array(json.loads(x)))
data['shape'] = data['image'].transform(lambda x: x.shape)
data['height'] = data['shape'].transform(lambda x: x[0])
data['width'] = data['shape'].transform(lambda x: x[1])
data['image_resized'] = data['image'].transform(lambda img: cv2.resize(img, dsize=(round(data['width'].mean()), round(data['height'].mean())), interpolation=cv2.INTER_LINEAR_EXACT))
data['image_resized_flattened'] = data['image_resized'].transform(lambda x: x.reshape((1, x.shape[0]*x.shape[1]))[0])

train_data, test_data = train_test_split(data, test_size=0.1)

X_training = train_data['image_resized_flattened'].apply(pd.Series)
y_training = train_data['name']
X_testing = test_data['image_resized_flattened'].apply(pd.Series)
y_testing = test_data['name']

model.fit(X_training, y_training)
predictions = model.predict(X_testing)

correct = 0
incorrect = 0
total = 0
for actual, predicted in zip(y_testing, predictions):
    total += 1
    if actual == predicted:
        correct += 1
    else:
        incorrect += 1

# Print results
print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")
