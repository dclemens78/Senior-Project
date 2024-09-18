# Danny Clemens, Devon Pedraza, Adam Boulos
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
parser.add_argument("-e", '--epoch', type=int, default=100, help="Set the number of epochs (iterations) during training")



def main(args): 

    train_loader, test_loader = load_data()
    
    model, criterion, optimizer, scheduler = create_network()
    
    train_cnn(model, train_loader, test_loader, criterion, optimizer, scheduler, args.epoch)
    
    if input("Would you like to see results? (Y/N): ").lower() == 'y': get_results()
        
    if args.debug: pdb.set_trace()


def create_network():
    
    class CNN(nn.Module):
        
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 1 input channel (grayscale), 16 output channels, 3x3 kernel
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # 16 inputs, 32 outputs, 3x3 kernel
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  
            
            self.pool = nn.MaxPool2d(2, 2)               # Pooling layer to reduce dimensionality
            self.fc1 = nn.Linear(64 * 16 * 16, 512)     # Adjust input size based on pooling
            self.fc2 = nn.Linear(512, 128)               # Fully connected layer
            self.fc3 = nn.Linear(128, 4)                 # Output layer for 4 classes
            
            # Adding dropout to reduce overfitting
            self.dropout = nn.Dropout(0.5)
            
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = torch.flatten(x, 1)                     # Flatten the dimensions
            x = self.dropout(F.relu(self.fc1(x)))       # Apply dropout
            x = self.dropout(F.relu(self.fc2(x)))       # Apply dropout
            x = self.fc3(x)
            return x

    model = CNN().to(DEVICE) # Set the model on our desired device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    return model, criterion, optimizer, scheduler
    

# Train the Neural Network
def train_cnn(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
    print("=============================================")
    print(f'Begining Training, Please Wait | Device : {DEVICE}')
    print("=============================================")
    
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
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) 
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
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
    print(f'Training Complete | Final Test Accuracy: {total_accuracy:.2f}%')
 

# Load the data
def load_data():
    ''' A method used to create and return our testing and training loaders '''
    
    # Transform: Resize, ToTensor, and Normalize
    transform = transforms.Compose([
        transforms.Resize((128, 128)),                             # Resize all images to 128x128
        transforms.Grayscale(num_output_channels=1),               # Convert to grayscale since MRI images are not in color
        transforms.RandomHorizontalFlip(p=0.5),                    # Randomly flip images horizontally
        transforms.RandomVerticalFlip(p=0.5),                      # Randomly flip images vertically
        transforms.RandomRotation(degrees=45),                     # Randomly rotate images within 45 degrees
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),       # Randomly crop and resize
        transforms.ColorJitter(brightness=0.2, contrast=0.2),      # Adjust brightness and contrast
        transforms.RandomAffine(degrees=0, shear=10),              # Apply affine transformations with shearing
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # Add slight perspective distortion
        transforms.ToTensor(),                                     # Convert to tensor for PyTorch
        transforms.Normalize(mean=[0.5], std=[0.5])                # Normalize based on grayscale range
    ])

    train_data = ImageFolder(TRAINPATH, transform=transform)
    test_data = ImageFolder(TESTPATH, transform=transform)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)  # Create data in batches
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    
    return train_loader, test_loader


# Display results
def get_results(): 
    ''' A method used to give the user a further understanding of their results via accuracy measures '''
    

if __name__ == "__main__": main(parser.parse_args())