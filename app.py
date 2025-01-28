import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os
from PIL import Image
import base64
from streamlit_autorefresh import st_autorefresh
import cv2
import numpy as np
import pyttsx3
import pickle
import csv

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function for speaking text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Define COL_NAMES globally
COL_NAMES = ['NAME', 'TIME']

# Function to mark attendance
def mark_attendance(user_name):
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

    # Check if the attendance CSV file for the current date exists
    exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

    # Create the attendance record
    attendance = [user_name, str(timestamp)]

    if exist:
        with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(attendance)
    else:
        with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(COL_NAMES)
            writer.writerow(attendance)

# Auto-refresh for attendance updates
count = st_autorefresh(interval=2000, limit=100, key="attendance_counter")

# Path for the attendance CSV file
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
attendance_file = f"Attendance/Attendance_{date}.csv"

# Ensure the Attendance folder exists
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

# Load user folders from the 'Users' directory
users_folder = "Users"
user_names = [name for name in os.listdir(users_folder) if os.path.isdir(os.path.join(users_folder, name))]

# Read existing attendance data if any
if os.path.exists(attendance_file):
    df = pd.read_csv(attendance_file, names=["NAME", "TIME"])
    present_users = df["NAME"].tolist()
else:
    df = pd.DataFrame(columns=["NAME", "TIME"])
    present_users = []

# Mark absent users
absent_users = [name for name in user_names if name not in present_users]

# Initialize a list to hold image data for displaying in the table
image_data = []

# Check user folders and process the images
if user_names:
    st.subheader("Captured Faces")
    for user_folder in user_names:
        user_folder_path = os.path.join(users_folder, user_folder)
        
        # Check if it's a directory (user folder)
        if os.path.isdir(user_folder_path):
            # List all images in the user's folder
            user_images = os.listdir(user_folder_path)
            
            # Select the first image (or modify this logic to pick the most suitable image)
            if user_images:
                image_path = os.path.join(user_folder_path, user_images[0])  # Choose the first image
                
                # Open and resize the image to create a thumbnail (e.g., 100x100)
                img = Image.open(image_path)
                img.thumbnail((100, 100))  # Resize to thumbnail size
                
                # Convert image to base64 so it can be displayed in Streamlit
                img.convert("RGB").save("temp_image.jpg")
                with open("temp_image.jpg", "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode()
                image_tag = f'<img src="data:image/jpeg;base64,{img_base64}" width="50" height="50"/>'
                
                # Find the timestamp for the user from the attendance file
                user_row = df[df["NAME"] == user_folder]
                if not user_row.empty:
                    attendance_time = user_row.iloc[-1]["TIME"]  # Get the last recorded timestamp
                else:
                    attendance_time = "Absent"

                # Prepare the data for displaying
                image_data.append({"NAME": user_folder, "TIME": attendance_time, "Image": image_tag})

else:
    st.write("No user folders found.")

# Display present users in a table
st.title("Attendance System")
st.subheader("Today's Attendance")
if not df.empty:
    st.write(df)

# Display absent users
st.subheader("Absent Users")
for user in absent_users:
    st.write(user)

# Display user images and attendance status in a table using HTML
st.subheader("User Images and Attendance Status")
if image_data:
    image_df = pd.DataFrame(image_data)
    st.write(image_df.to_html(escape=False), unsafe_allow_html=True)  # Allow HTML rendering to display images
else:
    st.write("No image data to display.")

# Error handling for face recognition model loading
try:
    with open('model_trained_knn.pkl', 'rb') as f:
        knn = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Face recognition model not found. Please train and load the model.")
    exit()

# Load label mapping
try:
    with open('data/names.pkl', 'rb') as f:
        label_mapping = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Label mapping file not found.")
    exit()

# Initialize video capture
video = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not video.isOpened():
    st.error("Error: Could not access the webcam.")
    exit()

facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Button to trigger face recognition and attendance marking
if st.button("Mark Attendance"):
    while True:
        ret, frame = video.read()

        # Check if frame is being captured correctly
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w, :]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(crop_img, (50, 50))
            resized_img = resized_img.flatten().reshape(1, -1)

            output = knn.predict(resized_img)
            user_name = label_mapping.get(output[0], "Unknown")

            if user_name != "Unknown":
                mark_attendance(user_name)
                speak(f"Attendance recorded for {user_name}")
                break

        # Exit the loop after one successful recognition
        if user_name != "Unknown":
            break

    video.release()