# coding: utf-8

# Face Recognition with OpenCV

# To detect faces, I will use the code from my previous article on [face detection](https://www.superdatascience.com/opencv-face-detection/). So if you have not read it, I encourage you to do so to understand how face detection works and its Python coding. 

# ### Import Required Modules

# Before starting the actual coding we need to import the required modules for coding. So let's import them first. 
# 
# - **cv2:** is _OpenCV_ module for Python which we will use for face detection and face recognition.
# - **os:** We will use this Python module to read our training directories and file names.
# - **numpy:** We will use this module to convert Python lists to numpy arrays as OpenCV face recognizers accept numpy arrays.

# In[1]:

#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as 
#it is needed by OpenCV face recognizers
import numpy as np

# Function to detect face using OpenCV
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('C:/Users/ayush/Downloads/opencv-face-recognition-python-master/opencv-face-recognition-python-master/opencv-files/lbpcascade_frontalface.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # If no faces are detected, return None
    if len(faces) == 0:
        return None, None

    # Assume only one face is detected, extract the face area
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

# Function to prepare training data
def prepare_training_data(data_folder_path, target_size=(100, 100)):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    
    # Iterate over each directory
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        label = int(dir_name.replace("s", ""))
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        subject_images_names = os.listdir(subject_dir_path)
        
        # Iterate over each image
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            image_path = os.path.join(subject_dir_path, image_name)
            image = cv2.imread(image_path)
            face, rect = detect_face(image)
            
            # Resize face to target size
            if face is not None:
                face_resized = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
                faces.append(face_resized)
                labels.append(label)
                
    return faces, labels

# Prepare training data
print("Preparing data...")
faces, labels = prepare_training_data("C:/Users/ayush/Downloads/opencv-face-recognition-python-master/opencv-face-recognition-python-master/training-data")
print("Data prepared")

# Print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# Create Fisher Face Recognizer
face_recognizer = cv2.face.FisherFaceRecognizer_create()

# Train the face recognizer
face_recognizer.train(faces, np.array(labels))

# Function to draw rectangle on image 
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
# Function to draw text on given image
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# Function to predict
def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    
    # Return None if no face detected
    if face is None:
        return None
    
    # Resize detected face to the same size as training images
    face_resized = cv2.resize(face, (100, 100), interpolation=cv2.INTER_AREA)
    
    label, confidence = face_recognizer.predict(face_resized)
    return label

# Prediction
print("Predicting images...")
test_img1 = cv2.imread("test5.jpg")
test_img2 = cv2.imread("test3.jpg")
predicted_label1 = predict(test_img1)
predicted_label2 = predict(test_img2)
print("Prediction complete")

# Get the name corresponding to the predicted label
subjects = ["", "Ayush Raj", "Sujan Reddy"]
name1 = subjects[predicted_label1]
name2 = subjects[predicted_label2]

# Draw rectangle and text on test images
if predicted_label1 is not None:
    draw_rectangle(test_img1, (0, 0, test_img1.shape[1], test_img1.shape[0]))
    draw_text(test_img1, name1, 10, 30)
    
if predicted_label2 is not None:
    draw_rectangle(test_img2, (0, 0, test_img2.shape[1], test_img2.shape[0]))
    draw_text(test_img2, name2, 10, 30)

# Display images
cv2.imshow("Test Image 1", test_img1)
cv2.imshow("Test Image 2", test_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
