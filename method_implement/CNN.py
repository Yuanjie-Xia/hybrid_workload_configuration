import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from error_calculation import median_relative_error


# Define a simple dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(256 * 7, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(1, -1)  # Adjust flattening dimensions
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def simple_cnn(scaled_df_merge, num):
    X = torch.tensor(scaled_df_merge.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(scaled_df_merge.iloc[:, -1].values, dtype=torch.float32)

    train_size = X.shape[1]
    train_ratio = float(train_size / X.shape[0])
    # Step 4: Split the data into training and testing sets

    for n in range(1, 7):
        error_set = []
        print("Sample size: " + str(n))
        for k in range(1, 10):
            print("Num of repeat experiments: " + str(k))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - n * train_ratio),
                                                                random_state=42 + k)
            # input_channels = X_train.shape[1]
            y_train = y_train.view(-1, 1)
            y_test = y_test.view(-1, 1)
            dataset = CustomDataset(X_train, y_train)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

            # Initialize the model, loss function, and optimizer
            model = SimpleCNN()
            criterion = nn.MSELoss()  # Assuming it's a regression problem, adjust for classification
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Training loop
            epochs = 10
            for epoch in range(epochs):
                for inputs, targets in dataloader:
                    optimizer.zero_grad()
                    outputs = model(inputs.unsqueeze(1))  # Adding an extra dimension for the channel
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

            test_dataset = CustomDataset(X_test, y_test)
            test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

            # Evaluate the model on the test set
            model.eval()
            with torch.no_grad():
                total_loss = 0
                predictions = torch.tensor([], dtype=torch.float32)
                targets = torch.tensor([], dtype=torch.float32)
                for inputs, labels in test_dataloader:
                    outputs = model(inputs.unsqueeze(1))
                    # Ensure labels have the same shape as outputs for each sample in the batch
                    labels = labels.unsqueeze(1)
                    predictions = torch.cat((predictions, outputs), dim=0)
                    targets = torch.cat((targets, labels), dim=0)
                    total_loss += criterion(outputs, labels).item()

                average_loss = total_loss / len(test_dataloader)
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}')
                median_relative_err = median_relative_error(predictions, targets)
                print(f'Epoch [{epoch + 1}/{epochs}], Median Relative Error: {median_relative_err:.4f}')

