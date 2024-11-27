# Logistic Regression Model for League of Legends Match Prediction

This document outlines the implementation of a logistic regression model using PyTorch to predict match outcomes in the League of Legends dataset. The model is trained with hyperparameter tuning to optimize performance.

## Overview
The primary objective is to build and evaluate a logistic regression model on a dataset, employing cross-entropy loss to train the model and analyzing feature importance to interpret the model's predictions.

## Steps to Implement the Model
### 1. **Data Preparation**
The dataset is loaded and split into training and testing sets:
```python
# Load the dataset
data = pd.read_csv('/path/to/league_of_legends_data_large.csv')

# Split data into features (X) and target (y)
X = data.drop('win', axis=1)
y = data['win']

# Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### 2. **Feature Scaling**
Standardize the features to improve model convergence:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 3. **Convert to PyTorch Tensors**
Convert the standardized data into PyTorch tensors:
```python
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
```

### 4. **Model Definition**
Create a simple logistic regression model using PyTorch:
```python
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
```

### 5. **Training and Evaluation**
Train the model and evaluate it with different learning rates:
```python
def train_and_evaluate_model(lr):
    model = LogisticRegressionModel(X_train_tensor.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_epochs = 1000

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = (test_outputs >= 0.5).float()
        test_accuracy = (test_predictions == y_test_tensor).float().mean().item() * 100

    print(f"Testing Accuracy for learning rate {lr}: {test_accuracy:.2f}%")
    return test_accuracy
```

### 6. **Feature Importance Analysis**
Evaluate and visualize feature importance based on model weights:
```python
def evaluate_feature_importance(model, feature_names):
    weights = model.linear.weight.data.numpy().flatten()
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': weights
    })
    feature_importance_df['Absolute_Importance'] = feature_importance_df['Importance'].abs()
    feature_importance_df = feature_importance_df.sort_values(by='Absolute_Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Absolute_Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Absolute Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
```

## Hyperparameter Tuning
The learning rates tested were: 0.01, 0.05, and 0.1. The best learning rate was chosen based on test accuracy.

## Results
```python
print("\nHyperparameter Tuning Results:")
for lr, accuracy in results.items():
    print(f"Learning Rate: {lr}, Test Accuracy: {accuracy:.2f}%")

print(f"\nBest Learning Rate: {best_lr} with Test Accuracy: {best_accuracy:.2f}%")
```

## Conclusion
This process shows how to implement and evaluate a logistic regression model using PyTorch, with steps for data preprocessing, training, evaluation, and feature analysis. The model's performance can be enhanced further with additional techniques such as regularization or hyperparameter tuning.

