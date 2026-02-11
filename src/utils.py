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

def build_cost_matrix(num_classes=10, default_wrong_cost=DEFAULT_WRONG_COST):
    # Initialize a 10x10 matrix with 0
    cost_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32)

    # Assign default cost to all incorrect predictions
    for true_label in range(num_classes):
        for pred_label in range(num_classes):
            if true_label != pred_label:
                cost_matrix[true_label, pred_label] = default_wrong_cost

    # function for readability
    def set_cost(true_label, pred_label, cost):
        cost_matrix[true_label, pred_label] = cost

    # Business-specific costly errors
    # Bag → Sneaker (Low)
    set_cost(true_label=8, pred_label=7, cost=COST_LOW)

    # Shirt → T-shirt (High)
    set_cost(true_label=6, pred_label=0, cost=COST_HIGH)

    # Coat → Pullover (High)
    set_cost(true_label=4, pred_label=2, cost=COST_HIGH)

    # Sandal → Sneaker (Medium)
    set_cost(true_label=5, pred_label=7, cost=COST_MED)

    # Ankle boot → Sneaker (Medium)
    set_cost(true_label=9, pred_label=7, cost=COST_MED)

    return cost_matrix

def get_test_predictions(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, labels in test_loader:
            X = X.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(X)

            # Predicted class
            preds = torch.argmax(outputs, dim=1)

            y_true.append(labels.cpu())
            y_pred.append(preds.cpu())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    return y_true, y_pred

def standard_accuracy(y_true, y_pred):
    return (y_true == y_pred).float().mean().item()

def cost_weighted_metrics(y_true, y_pred, cost_matrix):
    y_true = y_true.long()
    y_pred = y_pred.long()

    # Lookup cost per prediction
    per_item_cost = cost_matrix[y_true, y_pred]

    total_cost = per_item_cost.sum().item()
    avg_cost = per_item_cost.mean().item()

    max_cost = cost_matrix.max().item()

    # Normalize cost
    cost_weighted_accuracy = 1.0 - (avg_cost / max_cost)

    return total_cost, avg_cost, cost_weighted_accuracy

#Count misclassifications by (true, predicted) pair + their cost
def cost_breakdown(y_true, y_pred, cost_matrix, class_names, top_k=10):
    y_true = y_true.long()
    y_pred = y_pred.long()

    pair_counts = defaultdict(int)
    pair_total_cost = defaultdict(float)

    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t != p:
            c = float(cost_matrix[t, p].item())
            pair_counts[(t, p)] += 1
            pair_total_cost[(t, p)] += c

    # Sort by total cost descending
    ranked = sorted(pair_total_cost.items(), key=lambda x: x[1], reverse=True)

    print(f"Top {top_k} most costly misclassification pairs:\n")
    print(f"{'Rank':<5} {'True → Pred':<30} {'Count':<8} {'Cost/err':<10} {'Total cost':<10}")
    print("-" * 75)

    for i, ((t, p), total_c) in enumerate(ranked[:top_k], start=1):
        count = pair_counts[(t, p)]
        cost_per_error = total_c / count
        pair_name = f"{class_names[t]} → {class_names[p]}"
        print(f"{i:<5} {pair_name:<30} {count:<8} {cost_per_error:<10.2f} {total_c:<10.2f}")

    return ranked, pair_counts, pair_total_cost

def get_misclassified_examples(model, loader, device, max_examples=10):
    model.eval()
    misclassified = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            wrong_mask = preds != y
            if wrong_mask.any():
                X_wrong = X[wrong_mask].cpu()
                y_wrong = y[wrong_mask].cpu()
                preds_wrong = preds[wrong_mask].cpu()
                conf_wrong = probs[wrong_mask].max(dim=1).values.cpu()

                for i in range(X_wrong.size(0)):
                    misclassified.append({
                        "image": X_wrong[i],
                        "true": int(y_wrong[i]),
                        "pred": int(preds_wrong[i]),
                        "conf": float(conf_wrong[i])
                    })
                    if len(misclassified) >= max_examples:
                        return misclassified

    return misclassified

def plot_misclassified_examples(examples, categories):
    if len(examples) == 0:
        print("No misclassified examples found.")
        return

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(examples):
            ax.axis("off")
            continue

        img = examples[i]["image"].squeeze(0)
        true_label = examples[i]["true"]
        pred_label = examples[i]["pred"]
        conf = examples[i]["conf"]

        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(
            f"True: {categories[true_label]}\n"
            f"Pred: {categories[pred_label]} ({conf:.2f})",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()
