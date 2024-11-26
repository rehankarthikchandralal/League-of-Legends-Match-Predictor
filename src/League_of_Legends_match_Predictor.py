import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load the dataset
data = pd.read_csv('/home/rehan/Projects/League_of_Legends_match_Predictor/league_of_legends_data_large.csv')

# Split data into features (X) and target (y)
# 'win' is the target and the remaining columns are features
X = data.drop('win', axis=1)
y = data['win']

# Split the data into training and testing sets (80% for training, 20% for testing)
# random_state=42 ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing sets into PyTorch tensors
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Output the shapes of the resulting datasets for confirmation
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Convert scaled datasets to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # Convert to 2D tensor
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)    # Convert to 2D tensor

# Define the Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Linear layer for logistic regression
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Apply sigmoid activation

# Set input_dim
input_dim = X_train_tensor.shape[1]  # Number of features

# Initialize the Logistic Regression Model
model = LogisticRegressionModel(input_dim)

# Print the initialized model structure for confirmation
print("Initialized Model:")
print(model)

# Define the loss function
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent with learning rate 0.01

# Print loss and optimizer settings
print("\nLoss Function: Binary Cross-Entropy Loss (BCELoss)")
print("Optimizer: Stochastic Gradient Descent (SGD)")

# Additional Lines: Create a DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Batch size of 32
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Output the DataLoader structure for verification
print("\nDataLoaders:")
print(f"Train DataLoader: {len(train_loader)} batches")
print(f"Test DataLoader: {len(test_loader)} batches")
