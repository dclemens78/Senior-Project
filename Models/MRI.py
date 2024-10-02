# Danny Clemens
#
# MRI.py

''' A model that classifies brain scans in order to detect alzheimers disease. The results will be outputted on the website '''


import os
from efficientnet_pytorch import EfficientNet
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
import torch
from torch import nn
import torch.optim as optim
from collections import defaultdict
import pdb
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN, TEST = os.path.join(ROOT, 'Data', 'train'), os.path.join(ROOT, 'Data', 'test')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#class_counts = [717, 52, 2560, 1792]
#total_samples = sum(class_counts)
#weights = torch.tensor([total_samples / count for count in class_counts], dtype=torch.float).to(DEVICE)

def main():
    train_loader, val_loader, test_loader = load_images()

    model = build_model()

    model = model.to(DEVICE)

    train(model, train_loader, val_loader, num_epochs=100)
    
    test_accuracy = test(model, test_loader)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")



def load_images():
    # Augment the training data in order to better train the classifier
    train_transform = v2.Compose([
        v2.RandomResizedCrop([224, 224], scale=(0.8, 1.0), ratio=(0.75, 1.33), antialias=True),   
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(degrees=30), 
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        v2.ToImage(),  # Converts image to tensor (v2 equivalent of ToTensor)
        v2.ToDtype(torch.float, scale=True),  # Ensures the tensor is float32
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = v2.Compose([
        v2.Resize([224, 224], antialias=True),
        v2.ToImage(),  # Converts image to tensor (v2 equivalent of ToTensor)
        v2.ToDtype(torch.float, scale=True),  # Ensures the tensor is float32
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the full training dataset without any transformations
    full_train_dataset = ImageFolder(root=TRAIN)
    dataset_size = len(full_train_dataset)

    # Shuffle the indices
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    # Split into training and validation indices
    train_size = int(0.8 * dataset_size)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    # Use Subset to create the training and validation datasets
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)

    # Apply train transformations to the training set
    train_dataset.dataset.transform = train_transform

    # Apply validation/test transformations to the validation set
    val_dataset.dataset.transform = val_test_transform

    # Create data loaders for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Load test data using test transformations
    test_dataset = ImageFolder(root=TEST, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader



def build_model():
    ''' Construct the pretrained Efficientnet-B0 model '''
    
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_classes = 4 
    
    # Match the number of classes
    model._fc = nn.Sequential(
        
        nn.Linear(model._fc.in_features, num_classes)
    ) 
    return model


def train(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, patience=10):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.1)
    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        correct_per_class = defaultdict(int)
        total_per_class = defaultdict(int)
        
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
        
        # Update per-class tracking
        for label, prediction in zip(labels, predicted):
            total_per_class[label.item()] += 1
            if label == prediction:
                correct_per_class[label.item()] += 1
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.6f}, Accuracy: {accuracy:.2f}%')
        
        val_accuracy = validate(model, val_loader, criterion)
        print(f'Validation Accuracy: {val_accuracy:.2f}% | Epochs without improvement: {epochs_without_improvement}/{patience}')

        
        scheduler.step(val_accuracy)
        
        
        # Check for early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_without_improvement += 1
        
        # Stop training if patience is exceeded
        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        

def validate(model, val_loader, criterion):
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
    return accuracy


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                    
    accuracy = 100 * correct / total
    return accuracy



if __name__ == '__main__':
    main()
