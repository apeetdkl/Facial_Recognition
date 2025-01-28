import cv2
import os
import numpy as np
import time
import csv
from datetime import datetime
import pyttsx3  # For text-to-speech functionality
import pickle

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function for speaking text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Initialize the video capture, face detection, and face recognizer
video = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not video.isOpened():
    print("Error: Could not access the webcam.")
    exit()

facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load KNN model for face recognition
with open('model_trained_knn.pkl', 'rb') as f:
    knn = pickle.load(f)

# Load label mapping
with open('data/names.pkl', 'rb') as f:
    label_mapping = pickle.load(f)

# Background image for the window
imgBackground = cv2.imread("background.png")

# Column names for the attendance CSV
COL_NAMES = ['NAME', 'TIME']

# Track the last detected user to avoid multiple attendance records within a session
last_user = None

while True:
    ret, frame = video.read()

    # Check if frame is being captured correctly
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    print(f"Faces detected: {len(faces)}")  # Debugging line to check number of faces detected

    resized_frame = None  # Default value for resized_frame
    for (x, y, w, h) in faces:
        # Crop and resize the face
            crop_img = frame[y:y + h, x:x + w, :]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(crop_img, (50, 50))
            resized_img = resized_img.flatten().reshape(1, -1)

        # Predict the label for the current face
            output = knn.predict(resized_img)

        # Get the name of the recognized person
            user_name = label_mapping.get(output[0], "Unknown")

        # Only record attendance if the recognized user is different from the last one
        

            # Get timestamp for the attendance record
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

            # Check if the attendance CSV file for the current date exists
            exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

            # Draw the face bounding box and name label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, user_name, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

            # Create the attendance record
            attendance = [user_name, str(timestamp)]

            # Resize the frame to fit the background image
            resized_frame = cv2.resize(frame, (640, 480))  # Resize the frame
            # Only update background if resized_frame is available
            if resized_frame is not None:
                # Place the resized frame on the background
                imgBackground[162:162 + 480, 55:55 + 640] = resized_frame
                cv2.imshow("Frame", imgBackground)

            # Handle the 'o' key for recording attendance
            k = cv2.waitKey(1)
            if k == ord('o'):
                speak("Attendance Taken..")
                time.sleep(5)
                if exist:
                    with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(attendance)
                else:
                    with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(COL_NAMES)
                        writer.writerow(attendance)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close any open windows
video.release()
cv2.destroyAllWindows()