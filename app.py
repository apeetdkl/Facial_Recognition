import streamlit as st
import pandas as pd
import cv2
import numpy as np
import os
import time
import pickle
import csv
from datetime import datetime
from datetime import date
import pyttsx3  # For text-to-speech functionality
from PIL import Image
import shutil 
import threading
from streamlit.components.v1 import html
import base64
import json


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
        reg_date = date.today().strftime("%d-%m-%Y")
        user_data.append({"User Name": user, "Registration Date": reg_date})

# Show Registered Users Table and Add Download Button
if user_data:
    st.table(pd.DataFrame(user_data))  # Show registered users

    # Add a download button for registered users
    def convert_users_to_csv(user_data):
        df_users = pd.DataFrame(user_data)
        return df_users.to_csv(index=False).encode('utf-8')

    csv_users = convert_users_to_csv(user_data)
    st.download_button(
        label="Download Registered Users CSV",
        data=csv_users,
        file_name="Registered_Users.csv",
        mime="text/csv"
    )
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

    while count < 60:
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


def get_camera_component():
    return """
    <div>
        <video id="video" autoplay playsinline style="width:640px; height:480px;"></video>
        <canvas id="canvas" style="display:none;"></canvas>
    </div>
    <script>
        let video = document.getElementById('video');
        let canvas = document.getElementById('canvas');
        let stream = null;

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    video.play();
                };
                return true; // Camera started successfully
            } catch (error) {
                console.error("Error accessing camera:", error);
                return false; // Camera failed to start
            }
        }

        function captureFrame() {
            if (video.readyState === video.HAVE_ENOUGH_DATA) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                return canvas.toDataURL('image/jpeg', 0.8);
            }
            return null;
        }

        (async () => {  // Immediately invoked async function
            if (await startCamera()) { // Check if camera started
                window.addEventListener('message', (event) => {
                    if (event.data.type === 'capture_frame') {
                        const frame = captureFrame();
                        if (frame) {
                            window.streamlit.setComponentValue(frame);
                        }
                    }
                });
            } else {
                // Handle camera start failure (e.g., display a message)
                console.error("Camera failed to start. Check permissions and browser settings.");
                // You could also send a message to Streamlit to indicate failure.
            }
        })(); // Call the async function immediately
    </script>
    """

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
    # Create a container for the camera feed
    camera_container = st.empty()
    camera_container.html(get_camera_component())
    
    attended_users = {}
    running = True
    start_time = time.time()
    timeout = 30  # 30 seconds timeout
    
    while running and (time.time() - start_time < timeout):
        try:
            # Get frame data from the JavaScript component
            frame_data = st.session_state.get('camera_frame')
            if frame_data and frame_data.startswith('data:image/jpeg;base64,'):
                # Decode base64 image
                _, encoded = frame_data.split(",", 1)
                image_array = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Process frame for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = facedetect.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        crop_img = frame[y:y + h, x:x + w, :]
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                        resized_img = cv2.resize(crop_img, (50, 50))
                        resized_img = resized_img.flatten().reshape(1, -1)
                        
                        output = knn.predict(resized_img)
                        user_name = label_mapping.get(output[0], "Unknown")
                        
                        if user_name != "Unknown" and user_name not in attended_users:
                            if mark_attendance(user_name):
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                st.session_state.attendance_message = f"✅ Attendance marked for {user_name} at {timestamp}"
                                speak("Attendance Marked!")
                                attended_users[user_name] = True
                                running = False
                                break
            
            time.sleep(0.1)  # Short delay to prevent excessive CPU usage
            
        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            running = False
    
    # Clean up
    camera_container.empty()
    if not any(attended_users.values()):
        st.warning("No users were recognized during the attendance period.")
    
    st.rerun()  # Refresh the page to update attendance
    video.release()
    cv2.destroyAllWindows()
    

    


# Display the message (outside the button's if block)
if st.session_state.attendance_message:
    st.success(st.session_state.attendance_message)
    st.session_state.attendance_message = ""  # Clear the message
# ----------------------- Show Attendance Data (Corrected - Version 4) -----------------------
st.subheader("📅 View Attendance Records")

# Allow user to select a date
selected_date = st.date_input("Select Date:", datetime.now())
formatted_date = selected_date.strftime("%d-%m-%Y")
attendance_file = f"Attendance/Attendance_{formatted_date}.csv"

# Check if attendance file exists
if not os.path.exists(attendance_file):
    st.warning(f"⚠️ No attendance data available for {formatted_date}.")
else:
    # 1. Get all registered users (folders in "Users" directory)
    registered_users = set(os.listdir("Users"))

    # 2. Filter out deleted users (only existing folders)
    existing_registered_users = {user for user in registered_users if os.path.isdir(os.path.join("Users", user))}

    # 3. Initialize attendance data
    attendance_data = {}

    # 4. Read attendance file and mark "Present"
    df_attendance = pd.read_csv(attendance_file)
    for user in df_attendance["NAME"].values:
        if user in existing_registered_users:
            attendance_data[user] = "Present"

    # 5. Mark "Absent" for registered users with face data but not in attendance file
    for user in existing_registered_users:
        user_path = os.path.join("Users", user)
        has_face_data = any(file.endswith(('.jpg', '.png')) for file in os.listdir(user_path))

        if has_face_data and user not in attendance_data:
            attendance_data[user] = "Absent"

    # 6. Mark "No Face Data" for users without any images
    for user in existing_registered_users:
        user_path = os.path.join("Users", user)
        has_face_data = any(file.endswith(('.jpg', '.png')) for file in os.listdir(user_path))

        if not has_face_data and user not in attendance_data:
            attendance_data[user] = "No Face Data"

    # 7. Convert to DataFrame
    attendance_list = [{"NAME": user, "STATUS": status} for user, status in attendance_data.items()]
    df_final = pd.DataFrame(attendance_list)

    # Display attendance table
    if not df_final.empty:
        st.success(f"✅ Showing attendance for {formatted_date}")
        st.table(df_final)
    
    else:
        st.warning(f"⚠️ No attendance data available for {formatted_date}.")


            # Add a download button for attendance
        # Add a download button for attendance
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(df_final)
    st.download_button(
        label="Download Attendance CSV",
        data=csv,
        file_name=f"Attendance_{formatted_date}.csv",
        mime="text/csv"
    )

       
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