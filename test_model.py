import os
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

while True:
    model_name = input("Enter the name of the model file to load (e.g., 'mnist_model.pt'): ")
    
    if os.path.exists(model_name):
        break
    else:
        print(f"Model '{model_name}' not found. Please enter a valid model file.")

# Load the saved model
model = torch.load(model_name)
model.eval()  # Set model to evaluation mode

print(f"Model '{model_name}' successfully loaded.")

# Define transformations: Convert images to tensors and normalize them
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Define validation dataset and dataloader
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=False, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# Function to visualize predictions
def view_classify(img, ps):
    ''' Function for viewing an image and its predicted classes. '''
    ps = ps.cpu().data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze(), cmap='gray')
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

# Get a batch of validation images
images, labels = next(iter(valloader))

# Take the first image in the batch
img = images[0].view(1, 784)

# Disable gradients for evaluation
with torch.no_grad():
    logps = model(img)

# Convert log-probabilities to probabilities
ps = torch.exp(logps)

# Print the predicted digit
probab = list(ps.numpy()[0])

# View the image and predicted class
view_classify(img.view(1, 28, 28), ps)

# Evaluate the model accuracy
correct_count, all_count = 0, 0

for images, labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)
        
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        
        if(true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
