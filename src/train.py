import torch

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    # Initialize lists to track both training and validation metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    model.to(device)

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train() # Enable Dropout/BatchNorm
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Optimization steps
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        # Store average training metrics for the epoch
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)

        # --- VALIDATION PHASE ---
        model.eval() # Disable Dropout/BatchNorm for testing
        v_running_loss = 0.0
        v_correct = 0
        v_total = 0
        
        with torch.no_grad(): # Disable gradient tracking to save memory
            for v_batch_X, v_batch_y in test_loader:
                v_batch_X, v_batch_y = v_batch_X.to(device), v_batch_y.to(device)
                
                v_outputs = model(v_batch_X)
                v_loss = criterion(v_outputs, v_batch_y)
                
                v_running_loss += v_loss.item()
                _, v_predicted = torch.max(v_outputs.data, 1)
                v_total += v_batch_y.size(0)
                v_correct += (v_predicted == v_batch_y).sum().item()
        
        # Store average validation metrics for the epoch
        val_losses.append(v_running_loss / len(test_loader))
        val_accuracies.append(100 * v_correct / v_total)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}% | "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%")
    
    # Return all four lists for Deliverable #2 plotting
    return train_losses, val_losses, train_accuracies, val_accuracies