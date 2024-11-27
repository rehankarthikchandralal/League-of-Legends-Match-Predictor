import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('/home/rehan/Projects/League_of_Legends_match_Predictor/dataset/league_of_legends_data_large.csv')

# Split data into features (X) and target (y)
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

# Function to train and evaluate the model
def train_and_evaluate_model(lr):
    print(f"\nTraining with learning rate: {lr}")
    # Initialize the model
    model = LogisticRegressionModel(X_train_tensor.shape[1])

    # Define the loss function
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Batch size of 32
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Number of epochs
    num_epochs = 1000

    # Training loop
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

    # Evaluate the model
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Predictions on test data
        test_outputs = model(X_test_tensor)
        test_predictions = (test_outputs >= 0.5).float()
        test_accuracy = (test_predictions == y_test_tensor).float().mean().item() * 100

    print(f"Testing Accuracy for learning rate {lr}: {test_accuracy:.2f}%")

    # Evaluate Feature Importance
    evaluate_feature_importance(model, X_train.columns)

    return test_accuracy

# Function to evaluate feature importance
def evaluate_feature_importance(model, feature_names):
    # Extract weights from the trained model
    weights = model.linear.weight.data.numpy().flatten()

    # Create a DataFrame with feature names and their importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': weights
    })

    # Debug print to check the DataFrame content
    print("Feature Importance DataFrame:")
    print(feature_importance_df)

    # Sort features by the absolute value of their importance
    feature_importance_df['Absolute_Importance'] = feature_importance_df['Importance'].abs()
    feature_importance_df = feature_importance_df.sort_values(by='Absolute_Importance', ascending=False)

    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Absolute_Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Absolute Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    # Show the plot
    plt.show()


def save_predictions(model, X_test_tensor, y_test_tensor, filename='predictions.csv'):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = (test_outputs >= 0.5).float()

    predictions_df = pd.DataFrame({
        'Actual': y_test_tensor.squeeze().numpy(),
        'Predicted': test_predictions.squeeze().numpy()
    })

    predictions_df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

# Define learning rates to test
learning_rates = [0.01, 0.05, 0.1]

# Perform hyperparameter tuning
best_lr = None
best_accuracy = 0
results = {}

for lr in learning_rates:
    accuracy = train_and_evaluate_model(lr)
    results[lr] = accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_lr = lr

print("\nHyperparameter Tuning Results:")
for lr, accuracy in results.items():
    print(f"Learning Rate: {lr}, Test Accuracy: {accuracy:.2f}%")

print(f"\nBest Learning Rate: {best_lr} with Test Accuracy: {best_accuracy:.2f}%")
