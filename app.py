# Adam Boulos, Danny Clemens
#
# app.py

''' A program with the purpose of using Flask to connect our python ai script to our website '''

# PLEASE READ ALL COMMENTS 
import pdb   # pdb = python debugger. use the command pdb.set_trace() to stop your program at any point and print the values of any variable
from flask import Flask # Ignore the IDE warnings if they appear

APP = Flask(__name__)

# Note to Adam: pip install flask, or python -m pip install flask

def main():
    
    # I left a debugger in here, so play around with it. Type APPS and see what shows up. You can type any command. Type q to quit
    pdb.set_trace()
    print('Sucessfully Creted a Flask Application!!')

if __name__ == "__main__":
    main()
