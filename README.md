# Deep-Learning:

## RNN for Image Classification

### 📌 Project Overview

This project explores using a **Recurrent Neural Network (RNN)** for image classification, which is a unique approach since RNNs are typically used for sequence data. We treat images like sequences and apply RNNs to classify them.

### 📂 Dataset

* Custom or MNIST-like dataset
* Images are flattened and fed as sequences to the RNN.

### 🧠 Model Architecture

* Input images are reshaped into sequences
* RNN layer(s) used (e.g., `nn.RNN`, `nn.LSTM`)
* Fully connected output layer for classification

### 🚀 Training

* Loss Function: Cross-Entropy Loss
* Optimizer: Adam
* Epochs: \~10 (configurable)

### 📊 Evaluation

* Accuracy score
* Loss curve
* Confusion matrix (optional)

### 🔍 How to Run

1. Install required packages: `torch`, `matplotlib`, etc.
2. Run the notebook cell-by-cell

### 🎓 Learning Outcome

Great intro to sequence modeling with images and how RNNs can be repurposed creatively!

---

## CIFAR-10 Classification with Batch Normalization

### 📌 Project Overview

This project uses a Convolutional Neural Network (CNN) with **Batch Normalization** to improve training speed and performance on the **CIFAR-10** dataset.

### 📂 Dataset

* CIFAR-10 (10 classes, 32x32 RGB images)
* Loaded using torchvision

### 🧠 Model Architecture

* Multiple Conv2D layers
* BatchNorm2D applied after each convolution
* ReLU activations
* MaxPooling layers
* Fully connected layers for classification

### 🚀 Training

* Loss Function: Cross-Entropy Loss
* Optimizer: SGD or Adam
* Epochs: \~20 (configurable)

### 📊 Evaluation

* Training & Validation Accuracy
* Loss visualization
* Test set evaluation

### 🔍 How to Run

1. Install required packages: `torch`, `torchvision`, `matplotlib`
2. Run the notebook step-by-step

### 💡 Why BatchNorm?

* Stabilizes training
* Allows higher learning rates
* Reduces internal covariate shift

---

## Fashion-MNIST Classification 🧥👟

### 📌 Project Overview

This notebook focuses on classifying fashion items (like shirts, shoes, bags) using a **simple CNN model** trained on the **Fashion-MNIST** dataset.

### 📂 Dataset

* Fashion-MNIST (grayscale 28x28 images)
* 10 classes like T-shirt, trouser, etc.

### 🧠 Model Architecture

* Convolutional layers with ReLU
* MaxPooling for downsampling
* Fully connected layers
* Softmax output layer

### 🚀 Training

* Loss Function: Cross Entropy
* Optimizer: Adam
* Epochs: \~10

### 📊 Evaluation

* Accuracy on test set
* Visualization of sample predictions

### 🔍 How to Run

1. Install dependencies (`torch`, `torchvision`, etc.)
2. Launch notebook and execute all cells

### 🤖 What You Learn

* CNN basics
* Image classification pipeline
* Simple training loop and evaluation

---
