import streamlit as st
import pandas as pd
import cv2
import numpy as np
import os
import time
import pickle
import csv
from datetime import datetime
import pyttsx3  # For text-to-speech functionality
from PIL import Image
import shutil 
import threading

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function for speaking text
def speak(text):
    """Speaks text in a separate thread"""
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech).start()

   
# ----------------------- Initialize System -----------------------
st.title("Face Recognition Attendance System")
st.subheader("Registered Users")

# Ensure Attendance & Users folder exists
os.makedirs("Attendance", exist_ok=True)
os.makedirs("Users", exist_ok=True)

# ----------------------- Load face recognition model -----------------------
try:
    with open("model_trained_knn.pkl", "rb") as f:
        knn = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Face recognition model not found. Please train the model.")
    st.stop()

# Load label mapping
try:
    with open("data/names.pkl", "rb") as f:
        label_mapping = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Label mapping file not found.")
    st.stop()

# Load face detection model
facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

# ----------------------- Define retrain_model function -----------------------
from sklearn.neighbors import KNeighborsClassifier

def retrain_model():
    """Retrains the KNN model based on images in the Users directory."""
    facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    
    # Initialize KNN model
    knn = KNeighborsClassifier(n_neighbors=3)

    # Collect images and labels for training
    images = []
    labels = []
    label_mapping = {}
    current_label = 0

    # Directory containing user folders
    users_folder = "Users"

    # Iterate over each user in the Users directory
    for user in os.listdir(users_folder):
        user_folder = os.path.join(users_folder, user)
        if os.path.isdir(user_folder):
            # Process each image of the user
            for image_name in os.listdir(user_folder):
                if image_name.endswith(".jpg") or image_name.endswith(".png"):
                    image_path = os.path.join(user_folder, image_name)
                    image = cv2.imread(image_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    for (x, y, w, h) in faces:
                        face = gray[y:y+h, x:x+w]
                        face_resized = cv2.resize(face, (50, 50)).flatten()
                        images.append(face_resized)
                        labels.append(current_label)
            
            # Update label mapping with user name
            label_mapping[current_label] = user
            current_label += 1

    if len(images) > 0:
        # Convert images and labels to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        # Train the KNN model
        knn.fit(images, labels)

        # Save the trained KNN model
        with open("model_trained_knn.pkl", "wb") as f:
            pickle.dump(knn, f)
        
        # Save label mapping
        with open("data/names.pkl", "wb") as f:
            pickle.dump(label_mapping, f)

        print("Model retrained successfully!")
        return knn, label_mapping
    else:
        print("No images found for training!")
        return None, None

# ----------------------- Display Registered Users -----------------------
users_folder = "Users"
user_data = []

for user in os.listdir(users_folder):
    user_path = os.path.join(users_folder, user)
    if os.path.isdir(user_path):
        reg_date = time.strftime("%d-%m-%Y", time.gmtime(os.path.getctime(user_path)))
        user_data.append({"User Name": user, "Registration Date": reg_date})

if user_data:
    st.table(pd.DataFrame(user_data))  # Show registered users
else:
    st.warning("No users registered yet!")
# ----------------------- Add User Functionality -----------------------
if "add_user_message" not in st.session_state:
    st.session_state.add_user_message = ""


def add_user(user_name):
    """Captures face images, saves them, and retrains. Returns success status."""
    user_folder = os.path.join('Users', user_name)
    os.makedirs(user_folder, exist_ok=True)

    video = cv2.VideoCapture(0)
    count = 0
    stframe = st.empty()  # Placeholder

    while count < 50:
        ret, frame = video.read()
        if not ret:
            st.error("Failed to capture video frame.")
            return False  # Indicate failure

        frame = cv2.flip(frame, 1)  # Flip image for mirror effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue  # Skip if no faces are detected

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            file_path = os.path.join(user_folder, f"{user_name}_{count}.jpg")
            cv2.imwrite(file_path, face)
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        stframe.image(frame, channels="BGR", caption="Capturing Faces", use_container_width=True)

    video.release()
    cv2.destroyAllWindows()

    knn, label_mapping = retrain_model()
    if knn is not None and label_mapping is not None:
        st.session_state.add_user_message = f"User {user_name} added successfully and model retrained."
    else:
        st.session_state.add_user_message = f"User {user_name} added successfully, but model retraining failed or no images available for retraining."

    return True  # Indicate success


# Add User Form
with st.form(key='add_user_form'):
    new_user_name = st.text_input("Enter New User Name:")
    submit_button = st.form_submit_button("Add User")

    if submit_button and new_user_name:
        if add_user(new_user_name):
            st.rerun()  # Rerun after the form is submitted

if st.session_state.add_user_message:  # Display the message outside the form
    st.success(st.session_state.add_user_message)
    st.session_state.add_user_message = ""  # Clear the message
# ----------------------- Attendance Handling -----------------------
def mark_attendance(user_name):
    date = datetime.now().strftime("%d-%m-%Y")
    timestamp = datetime.now().strftime("%H:%M:%S")
    file_path = f"Attendance/Attendance_{date}.csv"

    if os.path.exists(file_path):
        if user_name in pd.read_csv(file_path)["NAME"].values: return False

    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)
        if not os.path.isfile(file_path): writer.writerow(["NAME", "TIME"])
        writer.writerow([user_name, timestamp])
    return True




# ----------------------- Attendance Handling -----------------------
# ----------------------- Attendance Handling (Full Code with File Clearing) -----------------------
def mark_attendance(user_name):  # Same as before
    date = datetime.now().strftime("%d-%m-%Y")
    timestamp = datetime.now().strftime("%H:%M:%S")
    file_path = f"Attendance/Attendance_{date}.csv"

    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["NAME", "TIME"])

        writer.writerow([user_name, timestamp])

    return True


if "attendance_message" not in st.session_state:  # Initialize session state
    st.session_state.attendance_message = ""

if st.button("Start Attendance", key="start_attendance_button"):
    video = cv2.VideoCapture(0)
    stframe = st.empty()
    attended_users = {}
    running = True
    start_time = time.time()
    timeout = 4

    while running:
        ret, frame = video.read()
        if not ret:
            st.error("Failed to capture frame.")
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

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, user_name, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

            if user_name != "Unknown" and user_name not in attended_users:
                if mark_attendance(user_name):
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.attendance_message = f"✅ Attendance marked for {user_name} at {timestamp}"  # Store in session state
                    speak("Attendance Marked!")
                    attended_users[user_name] = True
                   
                    break

        if any(attended_users.values()) and time.time() - start_time >= timeout: #Check timeout only if someone has been recognized
            running = False
            break

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            running = False

        stframe.image(frame, channels="BGR", use_container_width=True)

    video.release()
    cv2.destroyAllWindows()
    st.rerun()  # Refresh after attendance


# Display the message (outside the button's if block)
if st.session_state.attendance_message:
    st.success(st.session_state.attendance_message)
    st.session_state.attendance_message = ""  # Clear the message
# ----------------------- Show Attendance Data (Corrected - Version 4) -----------------------
st.subheader("Today's Attendance")
date = datetime.now().strftime("%d-%m-%Y")
attendance_file = f"Attendance/Attendance_{date}.csv"

# 1. Get *all* registered users
registered_users = set(os.listdir("Users"))  # Use a set for efficient lookup

# 2. Filter out deleted users (those whose folders no longer exist)
existing_registered_users = set()
for user in registered_users:
    user_path = os.path.join("Users", user)
    if os.path.isdir(user_path):  # Check if the folder still exists
        existing_registered_users.add(user)

# 3. Initialize attendance data (empty initially)
attendance_data = {}

# 4. Check attendance file and add present users (only for *existing* users)
if os.path.exists(attendance_file):
    df_attendance = pd.read_csv(attendance_file)
    for user in df_attendance["NAME"].values:
        if user in existing_registered_users:  # Check if user still exists
            attendance_data[user] = "Present"

# 5. Add registered users with faces who are NOT in the attendance file (Absent), but only if they exist
for user in existing_registered_users:
    user_path = os.path.join("Users", user)
    if any(file.endswith(('.jpg', '.png')) for file in os.listdir(user_path)):  # Check for face data
        if user not in attendance_data:  # Only add if NOT in CSV
            attendance_data[user] = "Absent"

# 6. Add users *without* face data, but only if they exist
for user in existing_registered_users:
    user_path = os.path.join("Users", user)
    if not any(file.endswith(('.jpg', '.png')) for file in os.listdir(user_path)):  # Check for NO face data
        if user not in attendance_data:
            attendance_data[user] = "No Face Data"


# 7. Convert to list of dictionaries
attendance_list = [{"NAME": user, "STATUS": status} for user, status in attendance_data.items()]

df_final = pd.DataFrame(attendance_list)
st.table(df_final)

# ----------------------- Delete User Functionality (modified) -----------------------
def delete_user(user_name):
    """Deletes a user, their images, updates the model, and removes attendance records."""
    user_folder = os.path.join("Users", user_name)
    if os.path.exists(user_folder):
        shutil.rmtree(user_folder)  # Remove the user's folder and images

        # Retrain the model after deleting the user
        knn, label_mapping = retrain_model()  # Retrain the model
        if knn is not None and label_mapping is not None: # Check if retrained
            st.success(f"User {user_name} and associated data deleted successfully. Model retrained.")
        else:
            st.warning(f"User {user_name} deleted but model retraining failed or no images available for retraining.")

        # Remove attendance records for the deleted user
        remove_attendance_records(user_name)  # Call the new function

        return True
    else:
        st.error(f"User {user_name} not found.")
        return False


def remove_attendance_records(user_name):
    """Removes attendance records for a specific user from all attendance files."""
    attendance_folder = "Attendance"
    for filename in os.listdir(attendance_folder):
        if filename.startswith("Attendance_") and filename.endswith(".csv"):
            filepath = os.path.join(attendance_folder, filename)
            try:
                df = pd.read_csv(filepath)
                if user_name in df["NAME"].values:  # Check if the user is in the file
                    df = df[df["NAME"] != user_name]  # Remove the user's records
                    df.to_csv(filepath, index=False)  # Save the updated CSV
                    print(f"Attendance records for {user_name} removed from {filename}")
            except pd.errors.EmptyDataError: # Handle the error if the file is empty
                print(f"Attendance file {filename} is empty.")
            except Exception as e: #Handle other exceptions
                print(f"An error occurred while processing {filename}: {e}")
    
# ----------------------- Delete User Form -----------------------
# ----------------------- Delete User Form (modified) -----------------------
st.subheader("Delete a User")
with st.form(key='delete_user_form'):  # Using a form for better Streamlit behavior
    users_folder = "Users"
    user_data = []

    for user in os.listdir(users_folder):
        user_path = os.path.join(users_folder, user)
        if os.path.isdir(user_path):
            reg_date = time.strftime("%d-%m-%Y", time.gmtime(os.path.getctime(user_path)))
            user_data.append({"User Name": user, "Registration Date": reg_date})

    user_names = [user["User Name"] for user in user_data]

    if user_names:
        selected_user = st.selectbox("Select User to Delete", options=user_names, key="delete_user_select")
    else:
        st.warning("No users available to delete.")
        selected_user = None

    submit_button = st.form_submit_button("Delete User")

    if submit_button:
        if selected_user:
            if delete_user(selected_user):
                st.rerun()  # Rerun to update the user list and attendance table
        else:
            st.error("Please select a user to delete.")