import os
import torch
from torchvision import transforms
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import numpy as np

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

# Define the image transformation (grayscale, resize, normalize)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28 pixels
    transforms.ToTensor(),  # Convert image to a tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to match the MNIST dataset format
])

# Function to process the image
def process_image(image_path):
    """Load and preprocess the image."""
    image = Image.open(image_path)  # Open the image
    image = transform(image)  # Apply transformations
    image = image.view(1, 784)  # Flatten to 784-long vector (1 x 28*28)
    return image

# Function to visualize the image and prediction probabilities
def view_classify(img, ps):
    """Display the image and predicted class probabilities."""
    ps = ps.cpu().data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze(), cmap='gray')
    ax1.axis('off')
    ax2.barh(range(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(range(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()

# Main function to run the script
def main(image_path):
    # Process the image
    img = process_image(image_path)

    # Predict the digit
    with torch.no_grad():  # Turn off gradients for faster inference
        logps = model(img)

    # Convert log probabilities to probabilities
    ps = torch.exp(logps)

    # Get the predicted digit and the probabilities
    probab = ps.numpy()[0]
    predicted_digit = np.argmax(probab)
    
    # Print the predicted digit and confidence levels for each digit
    print(f"Predicted Digit: {predicted_digit}")
    print("\nConfidence Levels for Each Digit:")
    for i, p in enumerate(probab):
        print(f"Digit {i}: {p * 100:.2f}%")
    
    # View the image and prediction probabilities
    view_classify(img.view(1, 28, 28), ps)

# Argument parser to handle image input from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an image with the trained "+model_name)
    parser.add_argument('image_path', type=str, help="Path to the input image file")
    args = parser.parse_args()

    main(args.image_path)
