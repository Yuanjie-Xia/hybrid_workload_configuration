import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from analysis_results import median_relative_error


class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionModel, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=4,  # Number of heads in the multiheadattention models
        )
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Ensure the input is 3-dimensional (batch_size, sequence_length, input_size)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        # Assuming you want to use the output from the last layer for the linear layer
        x = x[-1, :, :]

        # Apply a dense layer with activation function
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def focus_model(whole_data_df, exp_num):
    # Extract features and target variable
    X = whole_data_df.iloc[:, :-1].values
    y = whole_data_df.iloc[:, -1].values.reshape(-1, 1)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    train_size = X_tensor.shape[1]
    train_ratio = float(train_size / X_tensor.shape[0])

    for n in range(1, 7):
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=(1-n*train_ratio), random_state=42+exp_num)

        # Create DataLoader for training and testing
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Instantiate the model, loss function, and optimizer
        input_size = X.shape[1]
        output_size = 1
        hidden_size = 64  # Adjust as needed
        model = AttentionModel(input_size, hidden_size, output_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 40

        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                # Ensure labels have the same shape as outputs for each sample in the batch
                labels = labels.unsqueeze(1) # Add an extra dimension to match the output shape
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    total_loss = 0
                    predictions = torch.tensor([], dtype=torch.float32)
                    targets = torch.tensor([], dtype=torch.float32)
                    for inputs, labels in test_loader:
                        outputs = model(inputs)
                        # Ensure labels have the same shape as outputs for each sample in the batch
                        labels = labels.unsqueeze(1)
                        predictions = torch.cat((predictions, outputs), dim=0)
                        targets = torch.cat((targets, labels), dim=0)
                        total_loss += criterion(outputs, labels).item()

                    average_loss = total_loss / len(test_loader)
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')
                    median_relative_err = median_relative_error(predictions, targets)
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Median Relative Error: {median_relative_err:.4f}')