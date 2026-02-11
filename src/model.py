import torch
import torch.nn as nn

class FashionClassifier_Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # 28x28 -> 784
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # 10 clothing categories
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class FashionClassifier_2Hidden(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256) # Regularization: Batch Norm 
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.leaky_relu = nn.LeakyReLU(0.01) # it is known to perform better than ReLU
        # It provides a small value to the negatives, to prevent the dead neurons
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

class FashionClassifier_3Hidden(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.elu = nn.LeakyReLU(0.01) # it is known to perform better than ReLU
        # It provides a small value to the negatives, to prevent the dead neurons
        self.dropout = nn.Dropout(0.3) # Dropout 0.3 to prevent overfitting
        self.out = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.elu(self.fc1(x)))
        x = self.dropout(self.elu(self.fc2(x)))
        x = self.dropout(self.elu(self.fc3(x)))
        return self.out(x)