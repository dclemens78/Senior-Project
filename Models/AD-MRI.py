# Danny Clemens, Devon Pedraza, Adam Boulos
#
# AD-MRI.py
# (Alzheimer's Disease-Magnetic Resonance Imaging)

''' An AI Model that uses images of MRI brain scans to determine whether or not an individual has early onset Alzheimers disease '''

import numpy as np                            # Advanced math functions
import argparse                               # Make command line arguments
import pdb                                    # Debug the program
import torch                                  # PyTorch (used for AI)
from torch import nn                          # Neural Network?
from torchvision.datasets import ImageFolder  
from collections import Counter
import os                                     # To work with file paths
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F


# Command line arguments
parser = argparse.ArgumentParser(description="Use an Ai model to detect early onset Alzheimer's disease")
parser.add_argument('-d', '--debug', action='store_true', help='use pdb.set_trace() at end of program for debugging')
parser.add_argument('-r', '--results', action='store_true', help='display the results of the model')
parser.add_argument("-e", '--epoch', type=int, default=500, help="Set the number of epochs (iterations) during training")

# Conditional to determine if the computations should be run on the CPU or GPU
# Idea: add a command line argument to choose GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device selected: {device}')

dir_path = 'C:/Users/danie/Senior-Project/Models/Data'
test_ds = os.path.join(dir_path, 'test')
train_ds = os.path.join(dir_path, 'train')

# Transform: Resize, ToTensor, and Normalize
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming grayscale images
])

train = ImageFolder(train_ds)
test = ImageFolder(test_ds)

class_map = {0:'Mild Demented', 1:'Moderate Demented', 2:'Non Demented', 3:'Very Mild Demented'}
class_dict = dict(Counter(train.targets))
#class_dict

''' NOTES FOR OTHER COLLABORATORS:

1) convolutional neural network has been started, need to fill in the layers
2) after I determine the proper layers and their params, CNN will be constructed
3) I will then begin training
4) I will display results
5) Experiment with paralell programming

    Overall, CNN is pretty much made, I expect it to be complete by our next meeting. I also need to re-evaluate how we're taking in data,
    but this is good progress and I expect it to be complete within 2 weeks. It currently does not run because I have not defined the cnn layers.
    - Danny
'''



def main(args): 

    class CNN(nn.Module):
        
        def __init__(self):
            super().__init__()
            
            # Here, we will define each layer of our CNN
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 1 input channel, 16 output channels, 3x3 kernel
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 16 input, 32 output, 3x3 kernel
            self.pool = nn.MaxPool2d(2, 2)  # Pooling layer reduces dimensionality
            # Fully connected layers
            self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Adjust the input size to match the flattened feature map size
            self.fc2 = nn.Linear(128, 4)  # 4 output classes (Mild Demented, Moderate Demented, Non Demented, Very Mild Demented)
            
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # Flatten all dimensions except batch size
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = CNN().to(device) # Set the model on our desired device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    train_cnn(model, args)
    
    if input("Would you like to see results? (Y/N): ").lower() == 'y': get_results()
        
    if args.debug: pdb.set_trace()


# Train the Neural Network
def train_cnn(model, args):
    print(f'Begining Training, Please be Patient | Device : {device}')
    

def load_data():
    ''' A method used to load our testing and training data in numpy arrays '''
    pass


def get_results(): 
    ''' A method used to calculate the accuracy and confusion matrix of our model '''
    print('RESULTS:')


if __name__ == "__main__": main(parser.parse_args())