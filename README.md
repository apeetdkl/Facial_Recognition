Face Recognition Attendance System

Overview

This project is a Face Recognition Attendance System built using Streamlit, OpenCV, and Machine Learning (KNN). It allows users to:

Register users with face images.

Train and retrain a KNN-based face recognition model.

Mark attendance using face recognition.

View and manage attendance records.

Delete registered users and update the system accordingly.

Features

✅ User Registration – Capture face images and store them securely.
✅ Face Recognition – Identify users and mark attendance automatically.
✅ Attendance Tracking – Record and display attendance data.
✅ Model Retraining – Automatically update the recognition model when new users are added.
✅ User Deletion – Remove users and update records seamlessly.
✅ Speech Notification – Uses text-to-speech to confirm attendance marking.

Tech Stack

Programming Language: Python

Framework: Streamlit

Libraries Used:

OpenCV – For image processing and face detection.

NumPy & Pandas – For data handling.

pyttsx3 – For text-to-speech.

Scikit-learn – For KNN-based face recognition.

PIL – For image manipulation.

Installation & Setup

1️⃣ Install Dependencies

Ensure you have Python installed (>=3.7). Then, install the required libraries:

pip install streamlit opencv-python numpy pandas pyttsx3 scikit-learn pillow

2️⃣ Run the Application

Execute the following command:

streamlit run app.py

Usage

➤ Register a New User

Navigate to the "Registered Users" section.

Enter a new user name.

The system will capture 50 face images and store them.

The model will automatically retrain with the new user data.

➤ Mark Attendance

Click the Start Attendance button.

The webcam will recognize faces and mark attendance in a CSV file.

The system will announce attendance using text-to-speech.

➤ View Attendance

The "Today's Attendance" section displays the current day's attendance.

Attendance data is stored in the Attendance folder as CSV files.

➤ Delete a User

Select a user from the dropdown list.

Click Delete User to remove their data and retrain the model.

Attendance records related to the user will also be deleted.

Project Structure

📂 Face Recognition Attendance System
│── 📂 Users            # Stores user images
│── 📂 Attendance       # Stores attendance CSV files
│── 📂 data             # Stores model and label mapping
     │── haarcascade_frontalface_default.xml  # Face detection model
     │──faces_data.pkl #face datas for training
     │──names.pkl      #names for traning
│── app.py             # Main application script
│── model_trained_knn.pkl # Trained KNN model


Future Enhancements

📌 Implement Deep Learning (CNN) for improved accuracy.

📌 Add Multi-Camera Support for larger classrooms.

📌 Enhance UI/UX using Streamlit components.

📌 Export Attendance Data in different formats (Excel, PDF).

Author

👨‍💻 Developed by: [Your Name]📧 Contact: your.email@example.com

License

This project is licensed under the MIT License. Feel free to modify and improve!