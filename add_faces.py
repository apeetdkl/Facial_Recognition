import cv2
import os
import numpy as np
from PIL import Image
import pickle

# Initialize face detector and recognizer
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_trained.yml') 

# Initialize some parameters
user_name = input("Enter your name: ")

# Set the path to store the user images (in 'users' folder)
user_folder = os.path.join('users', user_name)  # Create a folder with the user's name inside 'users'

# Create user folder if doesn't exist
if not os.path.exists(user_folder):
    os.makedirs(user_folder)

# Open video capture
video = cv2.VideoCapture(0)

# Capture images for the user
count = 0
while count < 50:  # Capture 40 images for the user
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        continue  # Skip if no faces are detected
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        file_path = os.path.join(user_folder, f"{user_name}_{count}.jpg")
        cv2.imwrite(file_path, face)
        count += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Capturing Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print("Face images captured successfully.")

# Training the recognizer
print("Training the recognizer...")

face_samples = []
labels = []
label_mapping = {}  # Dictionary to store ID-to-name mapping
label_counter = 0

# Loop through users' folders and load images
for folder_name in os.listdir('users'):  # Loop through the 'users' folder
    folder_path = os.path.join('users', folder_name)
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image_name)
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            image_np = np.array(image, dtype=np.uint8)
            face_samples.append(image_np)
            labels.append(label_counter)

        # Add to label mapping (ID -> user name)
        label_mapping[label_counter] = folder_name
        label_counter += 1

# Train the recognizer with the collected data
recognizer.train(face_samples, np.array(labels))

# Save the trained model
recognizer.save("face_trained.yml")
print("Training complete. Model saved as 'face_trained.yml'.")

# Save faces and labels data
with open('data/faces_data.pkl', 'wb') as f:
    pickle.dump(face_samples, f)

with open('data/names.pkl', 'wb') as f:
    pickle.dump(label_mapping, f)

print("Faces data and labels saved successfully.")