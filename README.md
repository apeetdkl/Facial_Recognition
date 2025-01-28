# face_recognition_project
# Attendance System with Face Recognition

## Project Overview

This project is an **Attendance System** that utilizes **Face Recognition** to automatically mark the attendance of users based on their facial features. The system is built using **Streamlit**, **OpenCV**, and **Python**. It allows users to register their faces and automatically mark attendance when they are detected in front of the camera.

The project is designed to be user-friendly, real-time, and efficient for handling attendance in various environments like classrooms, offices, etc.

## Features

- **Face Registration**: Users can register their faces by adding their photos into the `Users` folder.
- **Automatic Attendance Marking**: When the user faces the camera, their face is detected and their attendance is automatically recorded.
- **Attendance Logging**: Attendance is recorded with timestamps and saved in CSV files for each day.
- **Text-to-Speech Integration**: When attendance is marked, the system will speak out the user's name as a confirmation.
- **User-Friendly Interface**: The attendance system has a simple web-based interface using Streamlit, making it easy to interact with and monitor attendance.
- **Live Face Recognition**: The system uses OpenCV's `Haar Cascade` for face detection and a KNN (K-Nearest Neighbors) model for recognition.

## Technologies Used

- **Streamlit**: For building the web application interface.
- **OpenCV**: For face detection and recognition.
- **Python**: The core programming language for this project.
- **pyttsx3**: For text-to-speech functionality.
- **Pickle**: For loading the pre-trained KNN model and label mapping.
- **Pandas**: For handling and displaying attendance data.
- **NumPy**: For numerical operations during image processing.

## Setup Instructions

To run the project locally, follow these steps:

### 1. Clone the Repository

```bash
git clone 
cd 