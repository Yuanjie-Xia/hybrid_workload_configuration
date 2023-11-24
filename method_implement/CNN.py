import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from error_calculation import mean_relative_error, median_relative_error


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
    def __init__(self, batch, x_shape, num_fc_layers):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(x_shape[0]*x_shape[1]*7, batch*4)
        self.fc_layers = nn.ModuleList([nn.Linear(batch*4, batch*4)])
        for _ in range(num_fc_layers - 1):
            # Dynamically add more fully connected layers
            self.fc_layers.append(nn.Linear(batch*4, batch*4))
        self.fc2 = nn.Linear(batch*4, batch)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(1, -1)  # Adjust flattening dimensions
        x = F.relu(self.fc1(x))
        for fc_layer in self.fc_layers:
            x = F.relu(fc_layer(x))
        x = self.fc2(x)
        return x


def simple_cnn(scaled_df_merge, num):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor(scaled_df_merge.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(scaled_df_merge.iloc[:, -1].values, dtype=torch.float32)

    train_size = X.shape[1]
    train_ratio = float(train_size / X.shape[0])

    for n in range(1, 7):
        error_set = []
        for k in range(1, 10):
            print("Num of repeat experiments: " + str(k))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - n * train_ratio),
                                                                random_state=42 + k)
    
            print("Sample size: " + str(len(y_train)))
            dataset = CustomDataset(X_train, y_train)
            b_size = int(len(y_train)/2)
            dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True)
            # Initialize the model, loss function, and optimizer
            model = SimpleCNN(b_size, X_train.shape, n).to(device)
            criterion = nn.MSELoss()  # Assuming it's a regression problem, adjust for classification
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Move the model to GPU
            model = model.to(device)

            # Training loop
            epochs = int(len(y_train))
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and targets to GPU
                    optimizer.zero_grad()
                    outputs = model(inputs.unsqueeze(1))
                    labels = labels.unsqueeze(1)
                    outputs = outputs.unsqueeze(1).view(b_size, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                average_loss = total_loss / len(dataloader)
                print(f'Training Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}')

            test_dataset = CustomDataset(X_test, y_test)
            test_dataloader = DataLoader(test_dataset, batch_size=b_size, shuffle=False)

            # Evaluate the model on the test set
            model.eval()
            with torch.no_grad():
                total_loss = 0
                pred = torch.tensor([], dtype=torch.float32, device=device)
                targ = torch.tensor([], dtype=torch.float32, device=device)
                for inputs, labels in test_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to GPU
                    try:
                        outputs = model(inputs.unsqueeze(1))
                        labels = labels.unsqueeze(1)
                        outputs = outputs.unsqueeze(1).view(b_size, 1)
                        pred = torch.cat((pred, outputs), dim=0)
                        targ = torch.cat((targ, labels), dim=0)
                        total_loss += criterion(outputs, labels).item()
                    except:
                        break

                average_loss = total_loss / len(test_dataloader)
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}')
                median_relative_err = median_relative_error(pred.to(device), targ.to(device))
                print(f'Epoch [{epoch + 1}/{epochs}], Median Relative Error: {median_relative_err:.4f}')
                mean_relative_err = mean_relative_error(pred.to(device), targ.to(device))
                print(f'Epoch [{epoch + 1}/{epochs}], Mean Relative Error: {mean_relative_err:.4f}')

