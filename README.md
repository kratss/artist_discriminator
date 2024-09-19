The StyleClassifier is an efficient image classification framework designed to distinguish between the distinctive artistic styles of Randall Munroe, creator of XKCD, and Nicholas Gurewitch, known for Perry Bible Fellowship. This project employs a streamlined Convolutional Neural Network (CNN) architecture implemented in PyTorch, ensuring rapid training times and a compact model size without compromising classification performance.

The repository includes a carefully curated dataset, straightforward preprocessing scripts, and user-friendly model training and evaluation pipelines. Comprehensive documentation is provided to facilitate reproducibility and encourage further exploration in the field of machine learning for art analysis. The StyleClassifier is an ideal resource for researchers and enthusiasts seeking a lightweight yet effective solution for style recognition in visual arts.

This repository includes:
- model implementation
- training scripts
- jupyter notebook for visualization
- dataset
- performance dataset

![Image Sample](/resources/sample_image.png)

# Overview

This repository contains a Convolutional Neural Network (CNN) designed to classify comic strips drawn by Randall Munroe (XKCD) and the artist behind Perry Bible Fellowship. The model leverages deep learning techniques to achieve high accuracy in distinguishing between these two distinct comic styles.

Table of Contents

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

    Convolutional Layers: Extracts features from comic images.
    Pooling Layers: Reduces dimensionality while retaining important information.
    Fully Connected Layers: Classifies the extracted features into the respective comic categories.

# Dataset

The dataset comprises a collection of drawings from both XKCD and Perry Bible Fellowship, sampled randomly from their respective archives. Each comic is labeled according to its source, allowing the model to learn the distinguishing features of each artist's style.

    Total Images: 117
    Traning: 907
    Test Set: 110

# Training Procedure

The model is trained using a simple yet effective approach, achieving satisfactory performance in just a few epochs. This rapid training process is a testament to the model's efficiency and the quality of the dataset.

    Epochs: 7
    Batch Size: 16
    Optimizer: Stochastic Gradient Descent
    Loss Function: Binary Cross-Entropy with Logits Loss

**Training Steps:**

    Load and preprocess the dataset.
    Define the CNN architecture.
    Compile the model with the chosen optimizer and loss function.
    Train the model on the training set while validating on the validation set.

**Results**

The model demonstrates solid performance in classifying the comic strips. Below are the performance metrics and visualizations of the results.
Performance Metrics

    Accuracy: 87%
    Loss: 0.01937
    Precision: [Insert precision value]
    Recall: [Insert recall value]

Charts

Training Loss Chart
Figure 1: Training Loss Over Epochs

Validation Accuracy Chart
Figure 2: Validation Accuracy Over Epochs
Usage

To use the model for classifying new comic strips, follow these steps:

    Clone the repository:

bash

git clone [repository-url]
cd [repository-directory]

Install the required dependencies:

bash

pip install -r requirements.txt

Run the classification script:

bash

    python classify_comic.py --image_path [path-to-comic-image]

Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, please open an issue or submit a pull request.
License

This project is licensed under the MIT License. See the LICENSE file for more details.

This README provides a comprehensive overview of the comic classification CNN project, highlighting its efficiency and effectiveness in a professional manner.
