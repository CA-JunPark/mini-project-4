import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

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
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
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

def analyze_confidence_thresholds(model, test_loader, device, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.5, 0.99, 50)
    
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Use Softmax to get probabilities (0.0 to 1.0)
            probs = F.softmax(outputs, dim=1)
            
            # Get the confidence (max probability) and the prediction
            confidences, predicted = torch.max(probs, 1)
            
            all_probs.extend(confidences.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy_scores = []
    acceptance_rates = []

    for t in thresholds:
        # Mask for items that meet the threshold
        mask = all_probs >= t
        
        # Percentage of items accepted
        accepted_count = np.sum(mask)
        acceptance_rate = (accepted_count / len(all_probs)) * 100
        
        # Accuracy on those accepted items
        if accepted_count > 0:
            acc = (all_preds[mask] == all_labels[mask]).mean() * 100
        else:
            acc = 0.0
            
        accuracy_scores.append(acc)
        acceptance_rates.append(acceptance_rate)
        
    return thresholds, accuracy_scores, acceptance_rates

def plot_confidence_analysis(thresholds, accuracy_scores, acceptance_rates):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Accuracy
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Accuracy on Accepted (%)')
    ax1.plot(thresholds, accuracy_scores, linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Acceptance Rate on the same X-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Items Accepted (%)')
    ax2.plot(thresholds, acceptance_rates, linewidth=2, linestyle='--', label='Acceptance Rate')
    ax2.tick_params(axis='y')

    plt.title('StyleSort: Accuracy vs. Acceptance Trade-off')
    
    # Save the deliverable
    plt.savefig('../results/confidence_threshold_analysis.png')
    plt.show()