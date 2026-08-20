


# CIFAR-10 Image Classification

CIFAR-10 is a dataset of 60,000 images of the following classes:  airplane, automobile, bird, cat, deer, dog, frogs, horses, ships, and trucks. Each of these classes will hold 6000 images per class. 
Can a machine learning model learn what a dog or a horse is based off the features of the object within the image? This project will explore this question by training a convolutional neural network (CNN) to classify images from the dataset. 
## Overview 

The goal is to build a CNN that is capable of learning the features of the class objects in the image and test the model's capabilities on classifying the object classes. The challenge with CIFAR-10, however, is that the images in the dataset is a 32x32 pixel image. A 32x32 image is very pixelated and may form difficulties in trying to classify the image even in a human perspective. Thus, can a CNN learn the visual features of the object classes and classify them correctly?
## Dataset

- 60,000 RGB images in the dataset
- 10 classes -  airplane, automobile, bird, cat, deer, dog, frogs, horses, ships, and trucks
- 40,000 training images 
- 10,000 validation images
- 10,000 test images

The dataset is split into 80% train set, 10% validation set, and 10% test set. 
The train set will be used to train the models and learn the features of the objects.
The validation set will be used to test the model's ability after training on whether or not the model has learned any features and if there are any signs of overfitting or underfitting. 
The test set will be used after training and validating to test the model's performance against images that have not been seen before. 

The model will run for a minimum of 30 epochs and a maximum of 50 epochs. Epochs is the iteration of the CNN learning the training images and then validating its performance.  After all 50 epochs are done, the CNN will be tested on the test set. 
## Model Architecture

The CNN takes a 32x32 RGB image as input and learns visual features through multiple convolutional layers. Batch normalization and ReLU activation are used after each convolutional layer, while max pooling reduces the spatial size of the feature maps. A dropout layer is also used before the final classification layer to help reduce overfitting.

The model architecture:

|Layer|Output|
|---|---|
|Input|3 x 32 x 32|
|Conv2D (3 → 16) + BatchNorm + ReLU|16 x 32 x 32|
|Conv2D (16 → 16) + BatchNorm + ReLU|16 x 32 x 32|
|MaxPool2D|16 x 16 x 16|
|Conv2D (16 → 32) + BatchNorm + ReLU|32 x 16 x 16|
|MaxPool2D|32 x 8 x 8|
|Conv2D (32 → 32) + BatchNorm + ReLU|32 x 8 x 8|
|Flatten|2048 features|
|Dropout|30%|
|Linear|10 output classes|

The final linear layer produces 10 prediction outputs corresponding to what the model predicts the object class t be. 
## Training 

- Optimizer: Adam
- Loss Function: Cross-Entropy Loss
- Training Batch Size: 64
- Validation Batch Size: 512
- Minimum Epochs: 30
- Maximum Epochs: 50
- Early Stopping Patience: 8 epochs
- Data Augmentation:
  - Random Horizontal Flip
  - Random Crop
- Learning Rate Scheduler: ReduceLROnPlateau
## Results

The model's accuracy for each class on the test set was:

| Class      | Accuracy |
| ---------- | -------: |
| Airplane   |    86.6% |
| Automobile |    90.0% |
| Bird       |    66.4% |
| Cat        |    57.4% |
| Deer       |    75.1% |
| Dog        |    73.6% |
| Frog       |    85.5% |
| Horse      |    80.7% |
| Ship       |    85.4% |
| Truck      |    83.1% |

## Training and Validation Accuracy/Loss


![[Pasted image 20260820085115.png]]

![[Pasted image 20260820085147.png]]

![[Pasted image 20260820085214.png]]

## Predictions

![[Pasted image 20260820085237.png]]


## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

```
python project.py
```


## License

This project is licensed under the MIT License.