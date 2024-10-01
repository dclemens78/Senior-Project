# Danny Clemens
#
# AD-MRI.py
# (Alzheimer's Disease-Magnetic Resonance Imaging)

''' An AI Model that uses images of MRI brain scans to determine whether or not an individual has early onset Alzheimers disease '''


import os                                    # Work with file paths                
import argparse                              # Create command line arguments
import pdb                                   # access the python debugger 
import torch                                 
import torch.optim as optim
from torch import nn                                     
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import ImageFolder 
from torchvision import models
from torch.utils.data import WeightedRandomSampler

import matplotlib.pyplot as plt
import random
from PIL import Image
from pathlib import Path
import shutil

""" 
Global Variables:
    ROOT:      the file path on the current device to this repository
    TESTPATH:  the file path to the testing images folder
    TRAINPATH: the file path to the training images folder
    DEVICE:    the device the program is running on (cpu or gpu)
"""

ROOT = os.path.dirname(os.path.abspath(__file__))
TESTPATH, TRAINPATH = os.path.join(ROOT, 'Data', 'test'), os.path.join(ROOT, 'Data', 'train')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f'Device selected: {DEVICE}')


# Command line arguments
parser = argparse.ArgumentParser(description="Use an Ai model to detect early onset Alzheimer's disease")
parser.add_argument('-d', '--debug', action='store_true', help='use pdb.set_trace() at end of program for debugging')
parser.add_argument('-r', '--results', action='store_true', help='display the results of the model')
parser.add_argument('-e', '--epoch', type=int, default=100, help="Set the number of epochs (iterations) during training")
parser.add_argument('-v', '--visualize', action='store_true', help="Visualize images from the dataset")



def main(args): 

    train_loader, test_loader = load_data(False)
    
    model, criterion, optimizer, scheduler = create_network()
    
    train_cnn(model, train_loader, test_loader, criterion, optimizer, scheduler, args.epoch)
        
    if args.debug: pdb.set_trace()


def create_network():
    model = models.resnet50(pretrained=True)  # Load the pre-trained ResNet-50 model
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)  # Adjust for 4 output classes
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    return model, criterion, optimizer, scheduler
    

# Train the Neural Network
def train_cnn(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
    print("==================================================")
    print(f'| Begining Training, Please Wait | Device : {DEVICE} |')
    print("==================================================")
    
    # Track the models total accuracy
    final_correct = 0
    final_total = 0
    
    for e in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Iterate over small batches of data
        # Inputs = images, labels = classes
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) # Move to same device as model
            
            optimizer.zero_grad()             # Clear the gradients
            outputs = model(inputs)           # Forward pass
            loss = criterion(outputs, labels) # Compute the loss
            loss.backward()                   # Backward pass (gradient calculation)
            optimizer.step()                  # Optimize the model

            running_loss += loss.item()  # Track the loss
            
            
        # Validation phase after each epoch
        model.eval()  # Set the model to evaluation mode
        test_loss = 0.0
        correct_per_class = {0: 0, 1: 0, 2: 0, 3: 0}  # Dictionary to store correct predictions for each class
        total_per_class = {0: 0, 1: 0, 2: 0, 3: 0}    # Dictionary to store total samples for each class
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) 
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                
                # Update per-class and overall statistics
                for label, prediction in zip(labels, predicted):
                    total_per_class[label.item()] += 1
                    if label == prediction:
                        correct_per_class[label.item()] += 1
                        
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        final_correct += correct
        final_total += total

        # Calculate average losses and accuracy
        train_loss = running_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)
        scheduler.step(test_loss)
        accuracy = 100 * correct / total
        
        # Print epoch results
        print(f'Epoch [{e+1}/{epochs}] Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Accuracy: {accuracy:.2f}%')

    # Print final total accuracy over all epochs
    total_accuracy = 100 * final_correct / final_total
    class_accuracies = {cls: 100 * correct_per_class[cls] / total_per_class[cls] for cls in correct_per_class if total_per_class[cls] > 0}
    print(f'Training Complete | Final Test Accuracy: {total_accuracy:.2f}%')
    print("Per-Class Accuracy:")
    for cls, accuracy in class_accuracies.items():
        print(f"Class {cls}: {accuracy:.2f}%")
 

# Load the data
def load_data(visualize):
    ''' A method used to create and return our testing and training loaders '''

    # Duplicate classes, as they are underrepresented
    duplicate_images(TRAINPATH, TRAINPATH, 'MildDemented', 1000)  # Duplicating MildDemented class
    duplicate_images(TRAINPATH, TRAINPATH, 'ModerateDemented', 2500)  # Duplicating ModerateDemented class
    
    # Transform: Resize, ToTensor, and Normalize
    transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Reduce image size to 128x128
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


    train_data = ImageFolder(TRAINPATH, transform=transform)
    test_data = ImageFolder(TESTPATH, transform=transform)
    

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)  # Create data in batches
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    if visualize == True:
        image_paths = []
        for path in train_data.imgs:
            image_paths.append(path[0])
            
        shuffled = image_paths.copy()    
        random.shuffle(shuffled)
        plt.figure(figsize=(5,5))
        for i, path in enumerate(shuffled):
            if i>3: break
            plt.subplot(2,2,i+1)
            img = Image.open(path)
            image_class = Path(path).parent.stem
            plt.title(image_class)
            plt.imshow(img)
        plt.tight_layout()
        plt.show()
    
    return train_loader, test_loader

# Duplicate images in the first and second classes, as they are severly underrepresented
def duplicate_images(source_dir, target_dir, class_name, num_duplicates):
    ''' Duplicate images from a given class to oversample '''
    class_path = Path(source_dir) / class_name
    target_path = Path(target_dir) / class_name
    
    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)
    
    images = list(class_path.glob('*.jpg'))  # Assuming images are in JPG format
    
    for i in range(num_duplicates):
        img_path = random.choice(images)  # Randomly choose an image to duplicate
        new_img_name = f"{img_path.stem}_dup{i}{img_path.suffix}"  # Give a new name
        shutil.copy(img_path, target_path / new_img_name)


# Display results
def test_single_image(model, image_path):
    """ Function to test a single image with the trained model """
    
    # Preprocessing pipeline (should match the one used during training)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Ensure model is in evaluation mode
    model.eval()

    # Move model and image to the same device
    image = image.to(DEVICE)
    model = model.to(DEVICE)
    
    # Disable gradient computation for inference
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)  # Get the index of the predicted class

    # Define the class labels (modify these to match your actual classes)
    classes = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    
    # Output the prediction result
    predicted_class = classes[predicted.item()]
    print(f'The model predicts that this MRI image shows: {predicted_class}')
    

if __name__ == "__main__": main(parser.parse_args())