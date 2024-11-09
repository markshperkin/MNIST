# MNIST Handwritten Digit Classification with PyTorch

This project implements a neural network using PyTorch to classify handwritten digits from the MNIST dataset. The code allows you to train a model, test it on the validation dataset, and even test the model on a specific image of your choosing.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Training the Model](#training-the-model)
   - [Testing the Model](#testing-the-model)
   - [Testing the Model with a Specific Image](#testing-the-model-with-a-specific-image)
4. [Source](#source)

## Requirements

Make sure you have the required dependencies listed in the `requirements.txt` file. These include essential libraries such as PyTorch, torchvision, and numpy.

### Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/markshperkin/MNIST.git
    ```

2. Navigate into the project directory:
    ```bash
    cd MNIST
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training the Model

To train the model, run the `train_model.py` script. This script will allow you to train the model on the MNIST dataset and save the trained model with a name of your choosing.

```bash
python train_model.py
```
### Training the Model with Adam Optimizer

To train the model with Adam optimizer, in train_model.py, change line 58 (' optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9) ') with:
```bash
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

### 2. Testing the Model

After training, you can test the model's performance on the MNIST validation dataset. Use the test_model.py script for this.

```bash
python test_model.py
```

The script will ask for the name of the model file that you want to test. Make sure to provide the correct filename (e.g., mnist_model.pt).

### 3. Testing the Model with a Specific Image

You can also test the model on a specific image by using the test_model_with_input.py script. This script allows you to pass an image file as an argument, and the model will attempt to classify the digit.

```bash
python test_model_with_input.py path/to/your/image.png
```
The program should convert your image to the required format.

## Source

This project is based on the tutorial provided by [Towards Data Science](https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627), which demonstrates how to build a neural network using PyTorch for digit classification.


