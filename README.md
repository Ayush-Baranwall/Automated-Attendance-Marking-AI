# Automated-Attendance-Marking-AI
## Automatic Attendance Marking
Face Recognition Attendance System

The Face Recognition Attendance System is a Python application that utilizes the DeepFace and RetinaFace libraries for face detection, recognition, and attendance marking. This system captures images from a camera, detects faces, compares them with an existing dataset, and marks attendance for recognized faces.
## Table of Contents

    Introduction
    Dependencies
    Installation
    Usage
    Contributing
    License

## Introduction

The Face Recognition Attendance System is designed to automate attendance marking using face recognition technology. It eliminates the need for manual attendance tracking, reduces errors, and improves efficiency. This system captures images from a camera, processes them to detect faces, compares the detected faces with an existing dataset, and marks attendance for recognized individuals.

The system utilizes the following libraries:

    OpenCV (cv2): Used for capturing images from the camera and image processing.
    Pandas (pd): Used for data manipulation and creating the attendance sheet.
    glob: Used for retrieving file paths.
    os: Used for file and directory operations.
    datetime: Used for generating the filename for the attendance sheet.
    deepface: Used for face recognition and verification.
    retinaface: Used for face detection.

## Dependencies

Before running the Face Recognition Attendance System, ensure that you have the following dependencies installed:

    Python 3.6+
    OpenCV (cv2)
    Pandas (pd)
    deepface
    retinaface

You can install the required dependencies using pip:

pip install opencv-python pandas deepface retinaface

Installation

To get started with the Face Recognition Attendance System, follow these steps:

 ### Clone the repository to your local machine:

git clone https://github.com/Ayush-Baranwall/Automated-Attendance-Marking-AI

 ### Change to the project directory:

cd automated attendance marking

 ### You can remove the commented lines that start with # to activate the corresponding functionality.

 ### Run the script:

python automated attendance marking.py

This will launch the system and start capturing images from the camera. Detected faces will be cropped and compared with the existing dataset for recognition. Attendance will be marked for recognized individuals.


    Connect a camera to your system or use the default system camera.

    Run the attendance_system.py script as mentioned in the installation steps.

    The system will access the camera and start capturing images frame by frame. It will save each frame as image.png.

    The captured image will be processed using RetinaFace to detect faces. Detected faces will be marked on the image.

    The marked faces will be cropped and saved as separate images for recognition.

    The system will compare the cropped faces with the existing dataset using DeepFace's verify function. Both VGG-Face and ArcFace models will be used for verification.

    If a face is recognized and verified, the corresponding name will be printed as "found [name]" in the console. The attendance will be marked as present for the recognized individual in the temporary attendance sheet.

    Once the recognition process is completed, an Excel file will be generated with the current date as the filename. This file will contain the attendance details, including the names of recognized individuals marked as present.

    The cropped images used for marking attendance will be removed to clear space for future images.

## Acknowledgements

This project utilizes the following libraries:

    DeepFace
    RetinaFace

A big thank you to the contributors and maintainers of these libraries for their valuable work.

Feel free to modify and expand this README file according to your specific needs. Include any additional sections, diagrams, or explanations that would help users understand and utilize your code effectively.
