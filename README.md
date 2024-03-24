# Facial Expression Detection using FER2013 Dataset

This repository contains a Jupyter notebook for building and training a Convolutional Neural Network (CNN) model to detect facial expressions using the FER2013 dataset.

## Implementation Steps

### 1. Environment Setup:

-   You will need a Python environment with necessary libraries installed. It is recommended to use Anaconda or a virtual environment for managing dependencies.

### 2. Import Necessary Libraries:

-   The notebook begins with importing libraries such as TensorFlow, Keras, OpenCV, etc., required for data processing, model building, and evaluation.

### 3. Dataset Collection:

-   The FER2013 dataset is used for training the facial expression detection model. This dataset contains grayscale images categorized into seven different emotions: angry, disgusted, fearful, happy, neutral, sad, and surprised.

### 4. Data Augmentation:

-   Data augmentation techniques such as rotation, horizontal flip, zoom, etc., are applied to increase the diversity of the training data and improve the model's generalization.

### 5. CNN Model Building:

-   A Convolutional Neural Network (CNN) architecture is designed and implemented for the facial expression detection task. The model architecture typically consists of multiple convolutional layers followed by pooling layers and fully connected layers.

### 6. Model Training:

-   The CNN model is trained on the augmented dataset using the Adam optimizer and categorical cross-entropy loss function. Training parameters such as batch size, number of epochs, learning rate, etc., were tuned for optimal performance.

### 7. Model Evaluation:

-   The trained model is evaluated on a separate test dataset to assess its performance in terms of accuracy. Additionally, confusion matrix, precision, recall, and F1-score metrics are computed to evaluate the model's performance on each emotion class.

## Performance

-   The performance of the trained facial expression detection model is reported as 62.91% accuracy on the test dataset. The achieved accuracy indicates the model's ability to correctly classify facial expressions.

## Model Details

-   The implemented model is a CNN architecture designed specifically for facial expression detection tasks. It leverages convolutional layers to extract features from input images and fully connected layers for classification.

## Remarks

The model's performance can be further improved through addressing the imbalance issue, experimenting with different advanced data augmentation, emsembling models, and experimenting with some hyperparameter tuning.
The next task would be to improve the model's performance by implementing some of these suggestions.


## Acknowledgements

-   The FER2013 dataset used in this project is publicly available and was originally published by Pierre-Luc Carrier and Aaron Courville. We acknowledge their contribution to the research community.

## References

-   Link to FER2013 dataset: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
-   Original paper: [FER2013 Dataset](https://arxiv.org/abs/1307.0414)


> Written with [StackEdit](https://stackedit.io/).
