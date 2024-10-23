# Adam Boulos
# app.py

''' A program that connects our Alzheimer's detection model with our website via FastAPI '''

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms
import io
from Models.MRI import build_model
import os
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Enable CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check device (GPU or CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Load trained model
model = build_model()
model.load_state_dict(torch.load(os.path.join('Models', 'best_model.pth'), map_location=DEVICE))
model = model.to(DEVICE)
model.eval()  # Set the model to evaluation mode

# Define the image transformations (same as in training)
transform = transforms.Compose([
    transforms.Resize([224, 224]),  # Resize to match model input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per model training
])

# Endpoint for image upload and model inference
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_bytes = await file.read()  
        image = Image.open(io.BytesIO(image_bytes)) 
        
        # Convert grayscale images to RGB 
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Transform the image for model input
        image = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension and send to the correct device

        # Run the model prediction
        with torch.no_grad():
            output = model(image)  # Get model output
            _, predicted = torch.max(output.data, 1)  # Get the index of the max log-probability (class prediction)

        # Map the predicted index to a label
        labels = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']
        predicted_label = labels[predicted.item()]  # Convert index to corresponding label

        return JSONResponse(content={"prediction": predicted_label})  # Return the prediction as JSON

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500) 

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
