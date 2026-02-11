import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def generate_confusion_matrix(model, test_loader, criterion, device, classes):
    model.eval() # Set to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculation to prevent weight updates
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store for Confusion Matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create the matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix')
    
    
    # Save to the results folder
    plt.savefig(f'../results/confusion_matrix.png')
    plt.show()

def plot_training_results(train_losses, val_losses, train_accs, val_accs):
    fig, axes = plt.subplots(1, 2)
    
    # Left plot: Training and Validation Loss
    axes[0].plot(train_losses, label='Train Loss', marker='o')
    axes[0].plot(val_losses, label='Val Loss', marker='o')
    # Updated title to match your PyTorch criterion
    axes[0].set_title('Cross Entropy Loss') 
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Right plot: Training and Validation Accuracy
    axes[1].plot(train_accs, label='Train Acc', marker='o')
    axes[1].plot(val_accs, label='Val Acc', marker='o')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    # Save to results folder - required deliverable
    plt.savefig(f'../results/training_curves.png')
    plt.show()
    
def plot_confusion_matrix(all_labels, all_preds, classes):
    # Create the matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrixss')
    
    
    # Save to the results folder
    plt.savefig(f'../results/confusion_matrix.png')
    plt.show()

def cost_weighted_accuracy(all_labels, all_preds):
    pass