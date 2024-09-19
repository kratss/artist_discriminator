![License](https://img.shields.io/badge/license-MIT-blue)
![Version](https://img.shields.io/badge/version-0.8-orange)

# Overview

This repository contains a Convolutional Neural Network (CNN) designed to identify art drawn by Randall Munroe of XKCD and the Nicholas Gurewitch. The model leverages deep learning techniques to achieve high accuracy in distinguishing between these two distinct comic styles.

The repository includes a carefully curated dataset, straightforward preprocessing scripts, and user-friendly model training and evaluation pipelines. Comprehensive documentation is provided to facilitate reproducibility and encourage further exploration in the field of machine learning for art analysis

![Image Sample](/resources/sample_image.png)

## Repository Contents
- Model implementation
- Training scripts
- Jupyter notebook for visualization
- Curated dataset
- Performance data 


## Table of Contents

Introduction

Model Architecture

Dataset

Training Procedure

Results

Usage

Contributing

License

# Introduction

Comic strips are a unique form of artistic expression, each with its own style and thematic elements. This project aims to utilize a CNN to automate the classification of comics, providing insights into the stylistic differences between the works of Randall Munroe, creator of [xkcd](xkcd.com) and artist [Nicholas Gurewitch](https://pbfcomics.com/).

# Model Architecture

The CNN architecture consists of several convolutional layers followed by pooling layers, culminating in fully connected layers for classification. The model is designed to be lightweight and efficient, allowing for rapid training and inference.

## Key Features:

- Convolutional Layers: Extracts features from comic images.
- Pooling Layers: Reduces dimensionality while retaining important information.
- Fully Connected Layers: Classifies the extracted features into the respective comic categories.

# Dataset

The dataset comprises a collection of drawings from both XKCD and Perry Bible Fellowship, sampled randomly from their respective archives. Each comic is labeled according to its source, allowing the model to learn the distinguishing features of each artist's style.

- Total Images: 117
- Training: 907
- Test Set: 110

# Training Procedure

The model is trained using a simple yet effective approach, achieving satisfactory performance in just a few epochs. This rapid training process is a testament to the model's efficiency and the quality of the dataset.

## Hyperparameters
- Epochs: 7
- Batch Size: 16
- Optimizer: Stochastic Gradient Descent
- Loss Function: Binary Cross-Entropy with Logits Loss

## Training Steps

1. Load and preprocess the dataset.
2. Define the CNN architecture.
3. Compile the model with the chosen optimizer and loss function.
4. Train the model on the training set while validating on the validation set.

# Results

The model demonstrates solid performance in classifying the comic strips. Below are the performance metrics and visualizations of the results.
Performance Metrics

    Accuracy: 87%
    Loss: 0.01937
    Precision: [Insert precision value]
    Recall: [Insert recall value]

**Charts**

Training Loss Chart
Figure 1: Training Loss Over Epochs

Validation Accuracy Chart
Figure 2: Validation Accuracy Over Epochs

# Usage

To use the model for classifying new comic strips, follow these steps:

Clone the repository:

`git clone https://github.com/kratss/artist_discriminator`

`cd artist_discriminator`

Install the required dependencies (available via pip, conda, and most distribution repositories):

`install matplotlib numpy pytorch`

Run the classification script:

`python classify_comic.py --image_path [path-to-comic-image]`

Reproduce the model with provided data or your own dataset:

- Find the original dataset in ./data/ or provide a custom dataset
- Point train.py to the location of your data
- Run python train.py
- Receive trained model in working directory

**Data Structure**
```
 data
 └─ comics
    ├─ artist_1
    │  ├── img1.jpg
    │  └── img2.jpg
    └─ artist_2
       ├── img1.jpg
       └── img2.jpg
```

# Contributing
Contributions are welcome! If you have suggestions for improvements or additional features, please open an issue or submit a pull request.

# License
This project is licensed under the MIT License. See the LICENSE file for more details.

