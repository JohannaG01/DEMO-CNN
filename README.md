# Convolutional Neural Network (CNN) for Image Classification

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The model architecture consists of multiple convolutional and pooling layers followed by dense layers for classification.

## Features

- Loads the CIFAR-10 dataset using TensorFlow/Keras.
- Defines a CNN model using TensorFlow/Keras.
- Trains the model on the training set.
- Evaluates the model on the test set.
- Makes predictions on a sample of test images and visualizes the predictions.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pillow (PIL)
- Colorlog (for colored logging output)

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:JohannaG01/DEMO-CNN.git
   cd DEMO-CNN
   ```

2. Create a virtual environment:

    <details>
    <summary>For Linux/MacOS</summary>

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    </details>
    <details>

    <summary>For Windows</summary>

    ```
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
    </details>

3. Install the required libraries:

    ```
    pip install -r requirements.txt
    ```

## Usage

Run the main script to train the model, evaluate it, and make predictions:

    
    python cnn.py
    

## Output

After running the script, the predictions will be saved as predictions.png in the current directory.
