import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # To avoid errors with matplotlib

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from time import time

# Define transformations: Convert images to tensors and normalize them
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Define training and validation datasets (with check if already exists)
trainset_path = 'PATH_TO_STORE_TRAINSET'
valset_path = 'PATH_TO_STORE_TESTSET'

if not os.path.exists(trainset_path):
    os.makedirs(trainset_path)

if not os.path.exists(valset_path):
    os.makedirs(valset_path)

trainset = datasets.MNIST(trainset_path, download=True, train=True, transform=transform)
valset = datasets.MNIST(valset_path, download=True, train=False, transform=transform)

# Define data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# Define the network architecture
input_size = 784  # 28x28 images flattened
hidden_sizes = [128, 64] # Hidden layers
output_size = 10  # 10 output neurons for digits 0-9

# Define each layer with its own activation function
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

print(model)

# Define loss function and optimizer
criterion = nn.NLLLoss()
# images, labels = next(iter(trainloader))
# images = images.view(images.shape[0], -1)  # This part of the code is redundant due to computing the loss for each batch of images after perfoming the forward pass at line 70

# logps = model(images) #log probabilities
# loss = criterion(logps, labels) #calculate the NLL loss

# Define optimizer who performs gradient descent and updates the weights over the training set
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

# Train the model
epochs = 15
time0 = time()
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784-long vector
        images = images.view(images.shape[0], -1)
    
        # Zero gradients before the pass
        optimizer.zero_grad()
        
        # Forward pass
        output = model(images)
        loss = criterion(output, labels)
        
        # Backpropagation
        loss.backward()
        
        # Optimize the weights
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Epoch {e+1} - Training loss: {running_loss/len(trainloader)}")

# Print training time
print("\nTraining Time (in minutes) =", (time()-time0)/60)

model_name = input("Enter a name for the model file (e.g., 'mnist_model.pt'): ")

# Save the trained model with the provided name
torch.save(model, model_name)

print(f"Model saved as {model_name}")

