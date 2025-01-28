import cv2
import os
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = []
labels = []
names = {}
id = 0

users_folder = "Users"
for user_folder in os.listdir(users_folder):
   user_folder_path = os.path.join(users_folder, user_folder)
   if os.path.isdir(user_folder_path):
       for image_file in os.listdir(user_folder_path):
           image_path = os.path.join(user_folder_path, image_file)
           img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
           if img is not None:
               faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
               for (x, y, w, h) in faces_detected:
                   face = img[y:y+h, x:x+w]
                   face_resized = cv2.resize(face, (50, 50))
                   faces.append(face_resized.flatten())
                   labels.append(id)
                   names[id] = user_folder
       id += 1

from sklearn.neighbors import KNeighborsClassifier
if len(faces) > 0 and len(labels) > 0:
   faces = np.array(faces)  # Convert list to numpy array
   labels = np.array(labels)
   knn = KNeighborsClassifier(n_neighbors=3)
   knn.fit(faces, labels)

   model_path = 'model_trained_knn.pkl'
   with open(model_path, 'wb') as f:
       pickle.dump(knn, f)

   with open('data/names.pkl', 'wb') as f:
       pickle.dump(names, f)
   print(f"KNN model trained and saved as '{model_path}'.")
else:
   print("No faces found for training. Please make sure you have images in the 'Users' directory.")