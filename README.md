# Handwritten Alphabet Recognition

![Handwritten Alphabet Recognition](https://static.hindawi.com/articles/cin/volume-2018/6747098/figures/6747098.fig.002.jpg)

## Overview

This project emphasizes the recognition of handwritten alphabets using a Convolutional Neural Network (CNN) model. Recognizing handwritten content is a challenging task due to the diversity and variability in human handwriting. To address this, we use a deep learning model to predict which alphabet a given handwritten image represents. With a vast dataset of grayscale images of handwritten alphabets, we aim to build a model that can accurately identify the handwritten characters, making it a valuable tool for applications such as automated form reading, document digitization, and more.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Model Building](#model-building)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset comprises 26 folders, each representing a letter from A-Z, with images of handwritten alphabets sized 28x28 pixels. Each alphabet in the image is center-fitted to a 20x20 pixel box and stored as Gray-level.

The dataset for this project can be accessed on Kaggle: [A_Z Handwritten Alphabets](https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format).

Before running the code, please download the dataset and place it in the project directory.

## Installation

1. Clone the repository:

```bash
git clone <https://github.com/qusaikarrar/Handwritten-Alphabet-Recognition.git>
cd <Handwritten-Alphabet-Recognition>
```

## Usage
Ensure the dataset is downloaded and present in the project directory.

Execute the Python script or Jupyter Notebook to train the model and view results.

## Data Preparation
Before feeding the data to our CNN model, it undergoes various preprocessing steps:

- **Loading Data:** The data is loaded into a pandas DataFrame.
- **Data Splitting:** The dataset is divided into features and labels, and further split into training and testing subsets.
- **Data Reshaping:** The data is reshaped to fit the requirements of a CNN model, with each image represented in 28x28x1 dimensions.
- **One-hot Encoding:** The labels are one-hot encoded to represent each of the 26 alphabets.

## Model Building
We use a CNN model comprising several layers:

- **Convolutional Layers:** Extract features from the images.
- **Pooling Layers:** Reduce the spatial dimensions while retaining the essential information.
- **Fully Connected Layers:** For classification purposes.

## Training
The model is compiled using the Adam optimizer and trained with a categorical cross-entropy loss function. Training metrics include accuracy and loss.

## Evaluation
The trained model's performance is evaluated using metrics such as training and validation accuracy and loss. Furthermore, predictions on test images are displayed to visualize the model's predictions against actual labels.

## Model Summary
The model comprises multiple layers:

- **Input Layer:** 28x28x1 (reflecting the image size and grayscale channel)
- **Convolutional and Pooling Layers:** Extract features from the images and reduce their dimensions.
- **Dense Layers:** Fully connected layers for classification.

**Total Parameters:** 137,178

For a detailed model summary, including layer shapes and parameters, refer to the provided code.

## Contributing
Feel free to contribute to this project. Any suggestions regarding improvements, additional features, or optimizations that could enhance the model's performance or usability are welcome.

## License
This project is free licensed.

