import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your DataFrame
# Assuming the DataFrame is named 'df'
# You might need to adjust the column indices based on your actual data
df = pd.read_csv('your_dataframe.csv', header=None)

# Extract features and target variable
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoader for training and testing
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the model with attention mechanism
class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionModel, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=4,  # Number of heads in the multiheadattention models
        )
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x[-1, :, :])  # Use the output from the last layer
        return x

# Instantiate the model, loss function, and optimizer
input_size = X.shape[1]
output_size = 1
hidden_size = 64  # Adjust as needed
model = AttentionModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()

    average_loss = total_loss / len(test_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

# Example: Make predictions on a new data point
new_data_point = torch.tensor([your_new_data_point], dtype=torch.float32)  # Replace with actual data
prediction = model(new_data_point)
print(f'Prediction: {prediction.item()}')
