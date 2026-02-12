# Mini Project IV – Deep Learning Classifier

## Business Problem Description 

The objective of this project is to build and evaluate a deep learning image classifier capable of accurately predicting handwritten digit classes (0–9). From a business perspective, this task simulates real world classification problems such as document digitization, automated form processing, and visual pattern recognition systems, where high accuracy is needed.

The business requirement for this project was to achieve at least 85% classification accuracy while exploring how depth and optimization strategies impact model performance and generalization.

## Dataset Description – Fashion-MNIST

**Fashion-MNIST** is a dataset of Zalando’s article images, designed as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It maintains the same image size, structure, and train–test split, while offering a more challenging and realistic classification task.

The dataset consists of:
- **60,000 training images**
- **10,000 test images**

Each example is a **28 × 28 grayscale image** associated with one of **10 clothing categories**.

### Data Structure

- Each image contains **784 pixels** (28 × 28).
- Each pixel has a single intensity value representing lightness or darkness.
- Pixel values range from **0 to 255**, where higher values indicate darker pixels.
- The training and test datasets contain **785 columns**:
  - The **first column** represents the class label.
  - The remaining **784 columns** represent pixel values.

To locate a pixel in the flattened representation, a pixel index `x` can be mapped back to the image using: 
`x+28i+j`

where:
- `i` is the row index (0–27)
- `j` is the column index (0–27)

For example, `pixel31` corresponds to the pixel located in the second row and fourth column of the image.

### Class Labels

| Label | Category       |
|------:|----------------|
| 0     | T-shirt / Top  |
| 1     | Trouser        |
| 2     | Pullover       |
| 3     | Dress          |
| 4     | Coat           |
| 5     | Sandal         |
| 6     | Shirt          |
| 7     | Sneaker        |
| 8     | Bag            |
| 9     | Ankle Boot     |

Fashion-MNIST preserves the simplicity of MNIST while providing more realistic visual complexity, making it well-suited for evaluating deep learning classifiers.

## Methodology

### Data Processing
- Images were transformed using `torchvision.transforms` to:
  - Convert images into tensors
  - Normalize pixel values to the range **[0, 1]**
- Data was loaded using PyTorch `DataLoader` with:
  - **Batch size = 32**
  - Shuffling enabled for training data

### Models Evaluated

1. **Baseline Model**
   - 1 hidden layer
   - ReLU activation
   - SGD optimizer

2. **2-Hidden Layer Model (Best Model)**
   - Batch Normalization
   - LeakyReLU activation
   - Adam optimizer

3. **3-Hidden Layer Model**
   - Dropout regularization
   - LeakyReLU activation
   - AdamW optimizer

## Model Architecture (Best Model)

The **2-Hidden Layer model** achieved the strongest balance between accuracy and stability.

### Input
- Flattened 28 × 28 image  
- **784 input features**

### Hidden Layers
- Layer 1: 784 → 256 neurons with Batch Normalization
- Layer 2: 256 → 128 neurons with Batch Normalization

### Activation Function
- **LeakyReLU**, used to reduce the risk of dead neurons and improve gradient flow

### Output Layer
- 128 → 10 neurons  
- Outputs raw logits for each class

### Loss Function
- **Cross-Entropy Loss**
  - Applies Softmax internally during training

### Optimizer
- **Adam optimizer** for adaptive and stable weight updates

## Results Summary with Key Metrics

The 2-Hidden Layer model achieved the highest overall performance and satisfied the business requirement of exceeding 85% accuracy.

- **Standard Test Accuracy:** **88.63%**
- **Cost-Weighted Accuracy:** **94.89%**

The improvement in cost-weighted accuracy indicates stronger performance when accounting for misclassification costs, which is particularly relevant in business settings where certain errors are more costly than others.

## Business Recommendations

- Deploy the **2-Hidden Layer architecture** for practical applications, as it offers the best balance between performance and complexity.
- Continue using **Batch Normalization and LeakyReLU** to improve training stability and convergence.
- Avoid unnecessary architectural depth, as the 3-Hidden Layer model increased complexity without meaningful performance gains.
- For future improvements:
  - Explore convolutional neural networks (CNNs) to better capture spatial features
  - Tune learning rates and batch sizes further
  - Adjust confidence thresholds to better align with specific business needs

## Setup instructions

    git clone <this repo>
    cd mini-project-4
    python -m venv .venv
    source .venv/Scripts/activate or .venv/bin/activate
    pip install -r requirements.txt

run .ipynb files with a created python environment.

## Team Member Contributions
Bryan Rachmat: Data analysis

Jun Park: Model training

Together: Learning Hub Report
