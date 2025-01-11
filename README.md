# Face-Recognition
Overview

This project implements a face recognition system using deep learning techniques. The system is built with TensorFlow and Kivy for creating an interactive application. It uses a Siamese neural network to verify if two images represent the same person.

Features

Face Verification: Matches a live webcam feed image against stored verification images.

Real-time Processing: Uses OpenCV for real-time image capture and preprocessing.

Custom Neural Network: Includes a custom L1 distance layer for calculating the similarity between embeddings.

Interactive GUI: Built with Kivy for user interaction.

Tech Stack

Python: Programming language used for development.

TensorFlow/Keras: Framework for building and training the neural network.

OpenCV: Library for image processing and video capture.

Kivy: Framework for building the graphical user interface.

Installation

Clone the repository:

git clone https://github.com/your-username/face-recognition.git

Navigate to the project directory:

cd face-recognition

Install the required dependencies:

pip install -r requirements.txt

Ensure TensorFlow is installed with GPU support (optional but recommended for faster processing).

Usage

Run the application:

python faceid.py

The application will open a GUI with a webcam feed and a "Verify" button.

To verify an identity:

Click the "Verify" button.

The system will compare the live image with stored verification images.

The verification result will be displayed on the screen with corresponding background color:

Green: Verified.

Red: Unverified.

White: Verification uninitiated.

Directory Structure

.
├── application_data
│   ├── input_image
│   └── verification_images
├── faceid.py
├── layers.py
├── requirements.txt
└── README.md

application_data: Stores input and verification images.

faceid.py: Main application file.

layers.py: Custom TensorFlow layers.

requirements.txt: List of required Python packages.

Model Details

Architecture: Siamese network with a custom L1 distance layer.

Training: Pre-trained model siamesemodelv2.h5 used for inference.

Preprocessing: Images resized to 100x100 and normalized to [0, 1].
