# Danny Clemens, Devon Pedraza, Adam Boulos
#
# AD-MRI.py
# (Alzheimer's Disease-Magnetic Resonance Imaging)

''' An AI Model that uses images of MRI brain scans to determine whether or not an individual has early onset Alzheimers disease '''

import numpy as np     # Advanced math functions
import argparse        # Make command line arguments
import pdb             # Debug the program
import torch                                  # PyTorch (used for AI)
#from torch import nn                          # Neural Network?
#from torchvision import ImageFolder
from collections import Counter
import os              # To work with file paths


# Command line arguments
parser = argparse.ArgumentParser(description="Use an Ai model to detect early onset Alzheimer's disease")
parser.add_argument('--debug', action='store_true', help='use pdb.set_trace() at end of program for debugging')
parser.add_argument('--results', action='store_true', help='display the results of the model')

# Conditional to determine if the computations should be run on the CPU or GPU
# Idea: add a command line argument to choose GPU or CPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dir_path = 'C:/Users/danie/Senior-Project/Models/Data'
test_ds = os.path.join(dir_path, 'test')
train_ds = os.path.join(dir_path, 'train')

#train = ImageFolder(train_ds)
#test = ImageFolder(test_ds)

#class_map = {0:'Mild Demented', 1:'Moderate Demented', 2:'Non Demented', 3:'Very Mild Demented'}
#class_dict = dict(Counter(train.targets))
#class_dict

def main(args): 
    
    
    if args.results:
        get_results()
        
    if args.debug:
        pdb.set_trace()
        

def load_data():
    ''' A method used to load our testing and training data in numpy arrays '''
    print('RESULTS:')


def get_results(): 
    ''' A method used to calculate the accuracy and confusion matrix of our model '''
    pass


if __name__ == "__main__":
    main(parser.parse_args())