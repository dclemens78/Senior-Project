# Danny Clemens, Devon Pedraza, Adam Boulos
#
# AD-MRI.py
# (Alzheimer's Disease-Magnetic Resonance Imaging)

''' An AI Model that uses images of MRI brain scans to determine whether or not an individual has early onset Alzheimers disease '''

import numpy as np
import argparse
import pdb
import keras

parser = argparse.ArgumentParser(description="Use an Ai model to detect early onset Alzheimer's disease")
parser.add_argument('--debug', action='store_true', help='use pdb.set_trace() at end of program for debugging')
parser.add_argument('--results', action='store_true', help='display the results of the model')

def main(args): 
    
    
    if args.results():
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