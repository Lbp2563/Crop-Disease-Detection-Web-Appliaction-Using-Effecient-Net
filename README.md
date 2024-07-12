# Crop Disease Detection Web Application Using EfficientNet

This repository contains a web application designed for detecting crop diseases using the EfficientNet model. The application is built with a focus on providing an easy-to-use interface for farmers and agricultural experts to identify and diagnose crop diseases from images.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Accurate and timely identification of crop diseases is crucial for ensuring food security and agricultural productivity. This application leverages the power of EfficientNet, a state-of-the-art convolutional neural network, to classify and diagnose various crop diseases from images uploaded by users.

## Features

- **Image Upload**: Users can upload images of crops for disease detection.
- **Disease Classification**: The application uses EfficientNet to classify the disease present in the uploaded image.
- **User-Friendly Interface**: A simple and intuitive web interface for easy navigation.
- **Scalable**: The application is designed to handle multiple concurrent users.
- **Real-time Results**: Fast and accurate disease detection results.

## Architecture

The architecture of the application is divided into the following components:

1. **Frontend**: Built with HTML, CSS, and JavaScript for a responsive and user-friendly interface.
2. **Backend**: Developed using Flask, a Python web framework, to handle image processing and model inference.
3. **Model**: EfficientNet model trained on a dataset of crop disease images.
4. **Storage**: Images and model data are stored and retrieved efficiently.

## Requirements

- Python 3.8 or later
- Flask
- EfficientNet
- TensorFlow or PyTorch (depending on the chosen framework)
- OpenCV
- Numpy
- Pandas

## Installation

Follow these steps to set up the application on your local machine:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Lbp2563/Crop-Disease-Detection-Web-Application-Using-Efficient-Net.git
    cd Crop-Disease-Detection-Web-Application-Using-Efficient-Net
    ```

2. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the pre-trained EfficientNet model** (if not included in the repository):
    - [EfficientNet Checkpoints](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

5. **Run the application**:
    ```bash
    python app.py
    ```

6. **Access the application**:
    Open a web browser and navigate to `http://localhost:5000`.

## Usage

1. **Upload Image**: Click on the 'Upload Image' button to select and upload an image of the crop.
2. **Get Results**: The application will process the image and display the disease classification results.
3. **View History**: Users can view previously uploaded images and their results.

## Model Training

To train the EfficientNet model on a custom dataset, follow these steps:

1. **Prepare Dataset**: Organize the dataset into training and validation sets. Ensure images are labeled correctly.
2. **Preprocess Data**: Use data augmentation techniques to enhance the dataset.
3. **Train Model**: Execute the training script provided in the repository.
    ```bash
    python train.py
    ```
4. **Evaluate Model**: Assess the model's performance on the validation set.
5. **Save Model**: Save the trained model for inference.

## Dataset

The dataset used for training the EfficientNet model should consist of images of crops with and without diseases. Each image should be labeled with the corresponding disease name.
Dataset Link: https://www.kaggle.com/datasets/emmarex/plantdisease

## Author

[Kajal Lochab](https://github.com/kajallochab)
[Lakshin Pathak](https://github.com/Lbp2563)
[Lakshit Pathak](https://github.com/Lakshit-25)
[Kapil Mehta]



## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to explore, use, and enhance this application to suit your needs. Happy coding!
