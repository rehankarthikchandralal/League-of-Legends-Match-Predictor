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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

# Set up the optimizer with L2 regularization (weight_decay)
# weight_decay = 0.01 applies L2 regularization
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

# Print loss and optimizer settings
print("\nLoss Function: Binary Cross-Entropy Loss (BCELoss)")
print("Optimizer: Stochastic Gradient Descent (SGD) with L2 Regularization (weight_decay=0.01)")

# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Batch size of 32
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Output the DataLoader structure for verification
print("\nDataLoaders:")
print(f"Train DataLoader: {len(train_loader)} batches")
print(f"Test DataLoader: {len(test_loader)} batches")

# Number of epochs
num_epochs = 1000

# Training loop with L2 Regularization
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()  # Clear gradients
        outputs = model(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        epoch_loss += loss.item()  # Accumulate batch loss

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# Model Evaluation
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient computation
    # Predictions on training data
    train_outputs = model(X_train_tensor)
    train_predictions = (train_outputs >= 0.5).float()  # Apply threshold for classification
    train_accuracy = (train_predictions == y_train_tensor).float().mean().item() * 100
    
    # Predictions on test data
    test_outputs = model(X_test_tensor)
    test_predictions = (test_outputs >= 0.5).float()
    test_accuracy = (test_predictions == y_test_tensor).float().mean().item() * 100

# Print accuracy after evaluation
print(f"\nTraining Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")
