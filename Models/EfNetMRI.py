# Danny Clemens
#
# EfNetMRI.py.py

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

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN, TEST = os.path.join(ROOT, 'Data', 'train'), os.path.join(ROOT, 'Data', 'test')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    train_loader, val_loader, test_loader = load_images()

    model = build_model()
    model = model.to(DEVICE)

    # Train the model and capture metrics for plotting
    training_stats = train(model, train_loader, val_loader, num_epochs=10)
    
    # Plot accuracy and loss graphs
    plot_metrics(training_stats)

    # Test the model
    test_accuracy, test_report, test_auc = test(model, test_loader)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print("Classification Report:\n", test_report)
    if test_auc:
        print(f"AUC Score: {test_auc:.2f}")
    
    # Generate a heatmap for a single sample (example usage)
    sample_input, sample_target = next(iter(test_loader))
    generate_heatmap(model, sample_input[0].unsqueeze(0).to(DEVICE), sample_target[0].item())

def load_images():
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
    ''' Construct the pretrained Efficientnet-B0 model '''
    
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_classes = 4  # We have four classes: No Impairment, Moderate Impairment, Mild Impairment, Very Mild Impairment
    
    model._fc = nn.Sequential(
        nn.Dropout(0.5),  # Dropout to reduce overfitting
        nn.Linear(model._fc.in_features, num_classes)
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

        # Save the best model if there's an improvement
        if improved:
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'Models/best_model.pth')
        else:
            epochs_without_improvement += 1

        # Early stopping if no improvement for 'patience' epochs
        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    return training_stats

def validate(model, val_loader, criterion):
    ''' Test (Validate) the Model on a Subset of Training Data that Seeks to Mimic Testing Data '''
    
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
    plt.legend()
    plt.show()

    # Plot Accuracy
    plt.figure()
    plt.plot(epochs, training_stats["train_accuracy"], label="Training Accuracy")
    plt.plot(epochs, training_stats["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.show()

def generate_heatmap(model, image_tensor, target_class):
    model.eval()
    layer_gc = LayerGradCam(model, model._blocks[-1])  # Target the last block of EfficientNet
    
    # Generate attribution
    attribution = layer_gc.attribute(image_tensor, target=target_class)
    
    # Convert attribution to a NumPy array and display with matplotlib
    attr_np = attribution.squeeze().cpu().detach().numpy()
    
    # Plot the heatmap
    plt.imshow(attr_np, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f"LayerGradCam - Target Class: {target_class}")
    plt.show()

if __name__ == '__main__':
    main()
