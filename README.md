# Cat and Dog Image Classification using CNN

## 📌 Project Overview

This project implements a Convolutional Neural Network (CNN) to classify images as either **Cat** or **Dog**. The model learns visual patterns such as shapes, textures, and features from labeled images and predicts the class of unseen images.

The project demonstrates the complete machine learning workflow, including data preprocessing, CNN model development, training, evaluation, and prediction.

## 🎯 Objective

The main objective of this project is to develop an image classification model capable of automatically distinguishing between cats and dogs using deep learning techniques.

## 📂 Dataset

The dataset contains:

- **12,499 Cat images**
- **12,499 Dog images**
- **Total: 24,998 images**

The images were divided into:

- Training set: 19,998 images
- Validation set: 2,500 images
- Test set: 2,500 images

## 🔧 Data Preprocessing

The following preprocessing steps were performed:

- Images converted to RGB format
- Images resized to **128 × 128 pixels**
- Pixel values normalized between **0 and 1**
- Cat images labeled as `0`
- Dog images labeled as `1`
- Dataset divided into training, validation, and testing sets

## 🧠 CNN Architecture

The CNN model consists of:

1. Convolutional Layer – 32 filters
2. Max Pooling Layer
3. Convolutional Layer – 64 filters
4. Max Pooling Layer
5. Convolutional Layer – 128 filters
6. Max Pooling Layer
7. Flatten Layer
8. Dense Layer – 128 neurons
9. Dropout Layer – 0.5
10. Output Layer – Sigmoid activation

The model was compiled using:

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Evaluation Metric: Accuracy

## 📊 Training

The model was trained for **15 epochs** with a batch size of **32**.

The model achieved approximately:

- Training Accuracy: **97%**
- Validation Accuracy: **87%**

The difference between training and validation accuracy indicates some degree of overfitting.

## 📈 Model Evaluation

The trained model was evaluated on **2,500 unseen test images**.

| Metric | Score |
|---|---:|
| Accuracy | **86.96%** |
| Precision | **87.68%** |
| Recall | **86.00%** |
| F1-Score | **86.83%** |

### Confusion Matrix

```text
[[1099  151]
 [ 175 1075]]