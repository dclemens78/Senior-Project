from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from Models.AD_MRI import create_network, test_single_image  # Import your model functions
import torch

# Initialize Flask app
app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')  # Path to the 'uploads' directory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Use uppercase 'UPLOAD_FOLDER' as Flask convention

# Load your trained model
model, criterion, optimizer, scheduler = create_network()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))  # Load model
model.eval()  # Set model to evaluation mode

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Home route to render the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if an image file is uploaded
    if 'filepath' not in request.files:
        return jsonify({'result': 'No file part'}), 400

    file = request.files['filepath']

    # DEBUG: Log the uploaded file details
    print(f"File uploaded: {file.filename}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")

    # Save the uploaded file directly without checking for file extension
    if file:
        filename = secure_filename(file.filename)  # Secure the filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # File save path

        try:
            file.save(filepath)  # Save the file
            print(f"File saved to: {filepath}")  # DEBUG: Log the saved file path
        except Exception as e:
            print(f"Error saving file: {e}")  # DEBUG: Log any errors during file saving
            return jsonify({'result': f"File saving failed: {str(e)}"}), 500

        # Process the image and make a prediction
        result = test_single_image(model, filepath)

        # Return the result as a JSON response
        return jsonify({'result': result})

    return jsonify({'result': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
