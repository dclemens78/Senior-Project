# Danny Clemens
#
# EfNetMRI.py

''' A model that classifies brain scans in order to detect Alzheimer's disease. The results will be outputted on the website '''

import os
from efficientnet_pytorch import EfficientNet
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, roc_auc_score
from captum.attr import LayerGradCam
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import argparse
import pdb
import time
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN, TEST = os.path.join(ROOT, 'Data', 'train'), os.path.join(ROOT, 'Data', 'test')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = 4 # We have four classes: No Impairment, Moderate Impairment, Mild Impairment, Very Mild Impairment
print(torch.cuda.get_device_name(0))

print(f'Device Selected for Training: {DEVICE}')

# Command line arguments
parser = argparse.ArgumentParser(description="Classify Alzheimer's disease via MRI scans of the brain")
parser.add_argument("-s", "--save", action="store_true", help='Save the current model')
parser.add_argument("-debug", action='store_true', help='Debug the program using pdb.set_trace()')
parser.add_argument("-e", "--epochs", type=int, default=10, help='Set the number of epochs for training')
parser.add_argument("-b", "--batch", type=int, default=64, help='Set the batch size for training and testing')
parser.add_argument("-p", "--patience", type=int, default=5, help="Set the patience for early stopping in training")
parser.add_argument("--plot", action='store_true', help="Plot useful metrics to display relevant model information")





def main(args):
    ''' Driver method '''
    
    # Gather and manipulate desired data (Brain MRI Images)
    train_loader, val_loader, test_loader = load_images(args)

    # Load the pretrained efficientnet-b0 convolutional neural network
    model = build_model() 
    model = model.to(DEVICE)


    print("Starting training...")
    train_start_time = time.time()
    
    # Train the model
    training_stats = train(model, train_loader, val_loader, args.epochs)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    print(f"Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes).")

    # show useful model data
    if args.plot: plot_metrics(training_stats)

    # Test the model
    print("Starting testing...")
    test_start_time = time.time()
    test_accuracy, test_report, test_auc = test(model, test_loader)
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    print(f"Testing completed in {test_time:.2f} seconds ({test_time/60:.2f} minutes).")
    
    # Print the final test accuracy, classification report, and auc score
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print("Classification Report:\n", test_report)
    if test_auc:
        print(f"AUC Score: {test_auc:.2f}")

    
    
    if args.debug: pdb.set_trace()
    
    if args.save:
        
        while True:
            choice = str(input("Would you like to overwrite the best model path? (Y/N): ")).strip().lower()
            
            if choice in {'y', 'yes', 'n', 'no'}:
                
                if choice in {'n', 'no'}:
                    current_time = time.strftime("%Y%m%d-%H%M%S") #unique tag (current time) to prevent duplicate file names
                    path = f'Models/Model-Paths/efnet_model{current_time}.pth'
                else:
                    path = 'Models/Model-Paths/best_efnet_model.pth'
                
                print(f'Model Successfully Saved as {path}')
                torch.save(model.state_dict(), path)  
                break
            else:
                print("Error Saving, Please Enter a Valid Response\n")

    
def load_images(args):
    ''' a method that manipulates and stores all image data as needed '''
    
    
    '''
    Data Augmentation:
    1). Efficient Net expects 224x224 images, so we resize every image.
    2). Parameters provided by PyTorch documentation (mean, std, scale, etc.).
    3). RandomResizedCrop (typical for MRI images).
    4). Random Rotation (create a more robust image to challenge the model).
    5). We finish by converting every image to a tensor.
    '''
    
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Shuffle the images to prevent patterns caused by the order of images
    full_train_dataset = ImageFolder(root=TRAIN)
    dataset_size = len(full_train_dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    
    # use 80% of the training images for training, 20% for validation
    train_size = int(0.8 * dataset_size)
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    
    test_dataset = ImageFolder(root=TEST, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    return train_loader, val_loader, test_loader


def build_model():
    ''' Construct the pretrained Efficientnet-B0 model '''
    
    # Load the pretrained efficentnetb0 cnn
    model = EfficientNet.from_pretrained('efficientnet-b0') 
    
    model._fc = nn.Sequential(
        nn.Dropout(0.5),  # Dropout layer to reduce overfitting
        nn.Linear(model._fc.in_features, CLASSES)
    )
    
    return model


def train(model, train_loader, val_loader, num_epochs):
    ''' Train the Alzheimer's Model to Accurately Predict Which Class an Image Belongs to '''
    
    learning_rate = 0.001
    patience = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)  # Scheduler to help the loss and accuracy properly converge

    # training statistics
    training_stats = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    # early stopping criteria
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader, start=1):
            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.2f}%')

        # Store training stats
        training_stats["train_loss"].append(train_loss)
        training_stats["train_accuracy"].append(train_accuracy)
        
        # Validate the model
        val_accuracy, val_loss = validate(model, val_loader, criterion)
        print(f'Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Store validation stats
        training_stats["val_loss"].append(val_loss)
        training_stats["val_accuracy"].append(val_accuracy)

        scheduler.step()

        # Check for improvement
        improved = False
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            improved = True
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True

        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # stop if epochs without improvement exceeds the patience
        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    return training_stats


def validate(model, val_loader, criterion):
    ''' Test (Validate) the Model on a Subset of Training Data that Seeks to Mimic Testing Data '''
    
    model.eval() # set model to evaluate
    correct = 0
    total = 0
    running_loss = 0.0
    
    # Test each image 
    with torch.no_grad(): 
        for inputs, labels in val_loader:
            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    
    return accuracy, running_loss / len(val_loader)


def test(model, test_loader):
    ''' A method used to evaluate the performance of the model the test dataset '''
    
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # calculate accuracy and generate the classification report
    accuracy = 100 * correct / total
    report = classification_report(all_labels, all_predictions, target_names=test_loader.dataset.classes)
    
    # Calculate AUC score
    auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr')
    
    return accuracy, report, auc


def plot_metrics(training_stats):
    ''' a method used to plot useful graphs to display model performance '''
    
    epochs = range(1, len(training_stats["train_loss"]) + 1)

    # Plot Loss vs Epochs
    plt.figure()
    plt.plot(epochs, training_stats["train_loss"], label="Training Loss")
    plt.plot(epochs, training_stats["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.show()

    # Plot Accuracy vs Epochs
    plt.figure()
    plt.plot(epochs, training_stats["train_accuracy"], label="Training Accuracy")
    plt.plot(epochs, training_stats["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.show()


def generate_heatmap(model, image_tensor, target_class):
    ''' Generates a heatmap to show what areas of an image the model focuses on the most '''
    
    model.eval()
    layer_gc = LayerGradCam(model, model._blocks[-1])  # Target the last block of EfficientNet
    
    # Generate attribution
    attribution = layer_gc.attribute(image_tensor, target=target_class)
    
    # Convert attribution to a NumPy array and display with matplotlib
    attr_np = attribution.squeeze().cpu().detach().numpy()
    
    # Plot the heatmap
    plt.imshow(attr_np, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Target Class: {target_class}")
    plt.show()
    
     
def test_single_image(model, image_path):
    ''' Test the model on a single image and print the prediction and probability. '''
    
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    # Define the transformation (resize, normalize) as done in val_test_transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension
    
    # Set the model to evaluation mode and make a prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]  # Convert to probabilities
        predicted_class = torch.argmax(output, dim=1).item()  # Get predicted class index
    

    class_names = ["No Impairment", "Moderate Impairment", "Mild Impairment", "Very Mild Impairment"]
    
    # Print the results
    print(f"Predicted Class: {class_names[predicted_class]}%")
    print("Class Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"Class: {class_names[i]}: {prob:.2f}%")
    

if __name__ == '__main__':
    main(parser.parse_args())