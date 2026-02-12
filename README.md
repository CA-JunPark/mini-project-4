# Mini Project IV – Deep Learning Classifier

## Business Problem Description 

## Methodology
Data Processing: used `torchvision.transforms` to convert images to tensors and normalize them in range [0, 1].
The data was loaded in batches of 32 using `DataLoader`

Models:
1. Baseline Model (1-Hidden Layer) using ReLU activation and SGD optimizer
2. 2-Hidden Layers using Batch Normalization and LeakyReLU and Adam optimizer
3. 3-Hidden Layers using Dropout and LeakyReLU and AdamW optimizer

## Model Architecture
Our best model was the 2-Hidden Layers.

Input: Flattened 28x28 image (784 features)

Hidden Layers:
    Layer 1: 784 -> 256 neurons with Batch Normalization 
    Layer 2: 256 -> 128 neurons with Batch Normalization 

Activation Function: LeakyReLU was used to prevent dead neurons

Output: 128 -> 10 neurons outputting raw logits

Loss Function: Cross-Entropy Loss was used to calculate the loss (It applies Softmax internally while training)

Optimizer: Adam optimizer was used to update the weights

## Result summary with Key Metrics
The 2-Hidden Layer model achieved the highest performance metrics, satisfying the business requirement of >85% accuracy.

    Standard Test Accuracy: 88.63%.

    Cost-Weighted Accuracy: 94.89%.

## Business Recommendations

## Setup instructions

    git clone <this repo>
    cd mini-project-4
    python -m venv .venv
    source .venv/Scripts/activate or .venv/bin/activate
    pip install -r requirements.txt

run .ipynb files with a created python environment.

## Team Member Contributions
Bryan Rachmat: 

Jun Park: 

Together: Learning Hub Report