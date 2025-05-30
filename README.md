# Digit Recognizer Web App

An end-to-end project that trains a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset and deploys it as an interactive web application using Streamlit. Users can draw a digit on a canvas, and the app will predict what digit it is in real time.

## Table of Contents

- [Digit Recognizer Web App](#digit-recognizer-web-app)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Running the Web App](#running-the-web-app)
  - [Project Structure](#project-structure)
  - [Further Enhancements](#further-enhancements)
  - [Acknowledgements](#acknowledgements)

## Project Overview

This project showcases an end-to-end deep learning workflow:

1. **Model Training:**  
   A CNN is built with TensorFlow Keras to classify MNIST digits. The model uses layers such as `Conv2D`, `MaxPooling2D`, and `Flatten` before predicting digits with fully connected layers.

2. **Model Deployment:**  
   Once the CNN is trained, the model is saved to disk and then reloaded by a Streamlit web app.

3. **Interactive Web App:**  
   The web interface utilizes `streamlit-drawable-canvas` to allow users to draw digits. The drawn image is preprocessed to match the model's input shape, and the app outputs the predicted digit.

## Features

- **Deep Learning with CNN:** Train a convolutional network on the MNIST dataset.
- **Model Serialization:** Save and reload the trained model.
- **Interactive UI:** Use Streamlit and a drawing canvas to capture user input.
- **Real-Time Prediction:** Preprocess and predict the drawn digit on the fly.
  
## Requirements

The project depends on the following Python libraries:

- TensorFlow (>= 2.10.0)
- Streamlit (>= 1.15.0)
- streamlit-drawable-canvas (>= 0.8.2)
- OpenCV-Python (>= 4.5.3.56)
- NumPy (>= 1.19.5)

See the [requirements.txt](./requirements.txt) included in the project.

## Installation

1. **Clone the Repository**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
2. **Set Up a Virtual Environment (Optional, but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate          # On Unix/macOS
   venv\Scripts\activate             # On Windows
3. **Install the Dependencies**
   ```bash
   pip install -r requirements.txt  
## Usage
1. Train and Save the Model
- Run the train_and_save.py script to build, train, and save the CNN model on the MNIST dataset. This script includes:
- Data loading and preprocessing (normalization and reshaping).
- Building a CNN model with convolutional and pooling layers.
- Training the model and evaluating its performance.
- Saving the trained model as mnist_cnn.h5.
  ```bash
  python train_and_save.py
You should see output detailing the training progress and a message confirming that the model was saved.

## Running the Web App
1. Launch the Streamlit App
Run the app.py script with Streamlit to start the interactive digit recognizer web app:
    ```bash
    streamlit run app.py
2. Interact with the App
- Open the provided URL (typically http://localhost:8501/) in your web browser.
- Use the drawing canvas to sketch a digit. The app will preprocess the drawing, load the saved model, and display the predicted digit along with a preview of the processed image.
  
## Project Structure
    ├── app.py                # Streamlit web application code
    ├── train_and_save.py     # CNN model training and saving script
    ├── requirements.txt      # List of required Python libraries
    ├── README.md             # Project documentation
    └── mnist_cnn.h5          # Saved Keras model (generated after training)

## Further Enhancements
- Model Improvements: Experiment with deeper architectures, dropout, or batch normalization.
- UI Enhancements: Improve the Streamlit interface with additional options like saving drawn digits, different color schemes, or adjusting stroke thickness.
- Deployment: Consider deploying the app to a cloud platform like Heroku, Streamlit Cloud, or Docker for broader accessibility.
- Additional Data: Use custom datasets or integrate real-time data streams for advanced recognition tasks.

## Acknowledgements
- Thanks to the TensorFlow, Keras, and Streamlit teams for providing the essential tools and frameworks.
- MNIST data courtesy of Yann LeCun.
- The streamlit-drawable-canvas component for offering a simple drawing interface in Streamlit.
