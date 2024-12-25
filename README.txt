Multi-Class Classification of Alzheimer’s Progression Stages

This project aims to detect Alzheimer's Disease at different stages using Convolutional Neural Networks (CNNs) and MRI images. The model is trained on MRI scans from the OASIS dataset and classifies images into four categories: Mild Dementia, Moderate Dementia, Very Mild Dementia, and Non-Demented.

Project Overview
The project involves building and training a deep learning model to classify MRI images of the brain into categories based on Alzheimer's Disease progression. We use multiple CNN architectures, including custom models and pre-trained models like VGG16, ResNet50, and ResNet152, to enhance classification accuracy.

Dataset
The dataset used for this project is from the OASIS (Open Access Series of Imaging Studies) dataset, which contains brain MRI scans of subjects with different stages of Alzheimer's Disease. The categories in the dataset are:

- Mild Dementia
- Moderate Dementia
- Very Mild Dementia
- Non-Demented

Project Structure
data_prep: Prepares the dataset by organizing images and labels.
plot_images: Displays sample images from each class for visualization.
CNN Models: Custom CNN model to detect Alzheimer's stages using convolutional layers and a fully connected layer.
Transfer Learning: Uses pre-trained models such as VGG16, ResNet50, and ResNet152 for improved model accuracy.
Dependencies

To run this project, ensure the following libraries are installed:

- tensorflow
- keras
- NumPy
- pandas
- OpenCV
- matplotlib
- sklearn

Usage
- Data Preparation: The images in the dataset are pre-processed for training and validation.
- Augmentation: Image augmentation techniques such as rescaling and random transformations are applied.
- Model Training: The model is trained using the ImageDataGenerator to load and preprocess the data in batches. The model is evaluated using the validation dataset.

Model Training
The project includes multiple models:
- Custom CNN Model: Three Layer and Five layer CNN models.
- VGG16 Model: A pre-trained VGG16 model is used for transfer learning by adding custom fully connected layers on top of the VGG16 base.
- ResNet50 Model: Similar to VGG16, but using the ResNet50 architecture.
- ResNet152 Model: The largest ResNet variant, used for enhanced performance.

Evaluation
The model's performance is evaluated using training and validation accuracy and loss graphs. The following metrics are plotted during training:
- Training Accuracy vs Validation Accuracy
- Training Loss vs Validation Loss

Results
Graphs displaying the training and validation accuracy/loss are provided after each model’s training. This gives insights into the model’s ability to generalize across different Alzheimer’s disease stages.
