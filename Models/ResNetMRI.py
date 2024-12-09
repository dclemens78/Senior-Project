# Danny Clemens
#
# ResNetMRI.py

''' A pretrained ResNet50 CNN to compete against our EfficientNet model '''

import os
from torchvision import models
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import time


# Set the global paths and device that the model will be trained on
ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN, TEST = os.path.join(ROOT, 'Data', 'train'), os.path.join(ROOT, 'Data', 'test')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = 4

def main():
    ''' Driver function of the program '''
    
    # Gather and manipulate the data (images) as needed
    train_loader, val_loader, test_loader = load_images()

    # Generate the ResNet50 model and 
    model = build_model()
    model = model.to(DEVICE)

    print("Starting training...")
    train_start_time = time.time()

    training_stats = train(model, train_loader, val_loader, num_epochs=10)
    
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    print(f"Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes).")
    
    # Plot the training/validation accuracy vs epoch and loss graphs
    plot_metrics(training_stats)

    # Test the model on the testing dataset and print detailed results
    test_accuracy, test_report, test_auc = test(model, test_loader)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print("Classification Report:\n", test_report)
    if test_auc:
        print(f"AUC Score: {test_auc:.2f}")  


def load_images():
    ''' Load the training and testing images into appropriate arrays and apply transformations as needed '''
    
    # Image transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    full_train_dataset = ImageFolder(root=TRAIN)
    dataset_size = len(full_train_dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    
    train_size = int(0.8 * dataset_size)
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    test_dataset = ImageFolder(root=TEST, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader

def build_model():
    ''' Construct the pretrained ResNet-50 model '''
    
   
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Dropout to reduce overfitting
        nn.Linear(model.fc.in_features, CLASSES)
    )
    
    return model

def train(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, patience=10):
    ''' Train the Alzheimer's Model to Accurately Predict Which Class an Image Belongs to '''
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)  # Scheduler to help the loss and accuracy properly converge

    # Store training statistics
    training_stats = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    # Early stopping flags
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Main training loop
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0 # track loss and training accuacy
        correct = 0
        total = 0
        
        # s = 1
        for i, (inputs, labels) in enumerate(train_loader):
            
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
        
        # Calculate training accuracy and loss for the given epoch
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader) 
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.2f}%')

        # Store training stats
        training_stats["train_loss"].append(train_loss)
        training_stats["train_accuracy"].append(train_accuracy)
        
        # Test the model on the validation set
        val_accuracy, val_loss = validate(model, val_loader, criterion)
        print(f'Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Store validation stats
        training_stats["val_loss"].append(val_loss)
        training_stats["val_accuracy"].append(val_accuracy)

        scheduler.step()

        # Early stopping
        # If the best loss and accuracy for validation do not improve for 10 epochs, stop early
        improved = False
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            improved = True
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True

        # Save the best model if there's an improvement
        if improved:
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'Models/best_model_resnet.pth')
        else:
            epochs_without_improvement += 1

        # Early stopping if no improvement for 'patience' epochs
        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    return training_stats

def validate(model, val_loader, criterion):
    ''' Test the model on a validation dataset during training '''
    
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
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
    ''' Test the model on a testing dataset and provide useful metrics '''
    
    model.eval() # Set model to evaluate
    correct = 0  # track test accuracy
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
    
    accuracy = 100 * correct / total
    report = classification_report(all_labels, all_predictions, target_names=test_loader.dataset.classes)
    
    # Calculate AUC score
    auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr') if len(np.unique(all_labels)) > 1 else None
    return accuracy, report, auc

def plot_metrics(training_stats):
    epochs = range(1, len(training_stats["train_loss"]) + 1)

    # Plot Loss
    plt.figure()
    plt.plot(epochs, training_stats["train_loss"], label="Training Loss")
    plt.plot(epochs, training_stats["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")


if __name__ == "__main__":
    main()
    
    

#mobilenetv2
#ResNet32