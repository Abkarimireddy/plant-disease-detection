"# Plant Disease Detection Project" 
# Crop Disease Detection System

## Overview
This project aims to develop a user-friendly and accessible crop disease identification system using Deep Learning and Python. The system will empower farmers to make informed decisions for improved crop health and agricultural productivity. It addresses several limitations of current deep learning-based crop disease identification systems, including limited and imbalanced datasets, overfitting, computational cost, and disease specificity.

## Current Limitations
- **Limited and Imbalanced Datasets**: Existing datasets may have a restricted number of crop types, diseases, and environmental variations, potentially leading to biased models.
- **Overfitting and Generalizability**: Complex models trained on limited data might struggle with unseen scenarios encountered in real-world fields.
- **Computational Cost and Accessibility**: Training deep learning models can require significant computational resources, limiting wider adoption.
- **Disease Specificity and User Interface**: Existing models might have difficulty distinguishing between visually similar diseases or may require complex interfaces.

## Proposed Solution
The project proposes a deep learning solution to overcome these limitations by focusing on:
- **Data Acquisition through Mobile App**: Develop a mobile application to collect high-quality images of crops (healthy and diseased) from farmers, contributing to a central database.
- **Data Augmentation and Preprocessing**: Use Python libraries to perform data augmentation (e.g., rotations, flips) and preprocessing (e.g., noise reduction, resizing) to create a robust dataset.
- **Transfer Learning and Efficient Model Design**: Employ pre-trained models like VGG16 or MobileNet and fine-tune them for crop disease classification, improving accuracy and reducing computational costs.
- **Multi-disease Detection and User Interface**: Create a model capable of identifying multiple diseases in a single image and design a user-friendly mobile app interface for capturing images, receiving diagnoses, and accessing treatment information.

## Methodology

### 1. Data Splitting
- Divide the dataset into training and test sets.

### 2. Preprocessing Layers
- Resize images to a uniform size.
- Normalize pixel values between 0 and 1.

### 3. CNN Architecture
- Stack convolutional layers with ReLU activation.
- Apply max-pooling layers.
- Flatten feature maps and add fully connected dense layers.
- Include an output layer with softmax activation for classification.

### 4. Training and Validation
- Train the CNN model using batches of preprocessed images.
- Use backpropagation and validation data to minimize loss and monitor for overfitting.

### 5. Evaluation and Metrics
- Evaluate model performance with accuracy, precision, recall, F1 score, and specificity.
- Visualize training and validation curves.

### 6. Probability Prediction
- Generate and visualize probability predictions for test images.

### 7. Batch Prediction
- Make and visualize batch predictions.

### 8. F1 Score Calculation
- Compute the F1 score to assess model performance.

## Experimental Setup

### Objective
Develop a deep learning model for accurate crop disease classification using TensorFlow and Keras.

### Dataset Description
- **Source**: PlantVillage dataset.
- **Total Images**: 20638 files 
- **Classes**: 15.
- **Preprocessing**: Images resized to 256x256 pixels and normalized.

### Model Architecture
- **Type**: Convolutional Neural Network (CNN).
- **Layers**: Convolutional layers, max-pooling, fully connected dense layers, softmax output.

### Training Procedure
- **Optimizer**: Adam
- **Epochs**: 40
- **Batch Size**: 32
- **Augmentation**: Random flips and rotations.

### Evaluation Metrics
- Accuracy, precision, recall, F1 score, specificity.

### Validation and Testing
- Dataset split: 80% training, 10% validation, 10% testing.
- Evaluate against baseline models.

### Reproducibility
- The code, dataset, and experimental settings are publicly available for reproducibility.

## Contact
For questions or support, please contact abikarimireddy@gmail.com or open an issue on GitHub.


