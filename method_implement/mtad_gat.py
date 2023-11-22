import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from mtad_model import mtad_gat_model
from mtad_model.training import *
from mtad_model.utils import create_data_loaders, SlidingWindowDataset


def mtad_impl(whole_data_df, exp_num):
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
        train_dataset = SlidingWindowDataset(X_train, X_train.shape[0], None)
        test_dataset = SlidingWindowDataset(X_test, X_train.shape[0], None)
        batch_size = 256
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, batch_size, 0.1, True, test_dataset=test_dataset
        )

        model = mtad_gat_model.MTAD_GAT(
            X_train.shape[1],
            X_train.shape[0],
            1,
            kernel_size=2
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        forecast_criterion = nn.MSELoss()
        recon_criterion = nn.MSELoss()
        trainer = Trainer(
            model,
            optimizer,
            0,
            X_train.shape[0],
            None,
            30,
            batch_size,
            0.001,
            forecast_criterion,
            recon_criterion,
            False,
            './',
            './',
            1,
            True,
            ""
        )
        trainer.fit(train_loader, val_loader)

        # Check test loss
        test_loss = trainer.evaluate(test_loader)
        print(f"Test forecast loss: {test_loss[0]:.5f}")
        print(f"Test reconstruction loss: {test_loss[1]:.5f}")
        print(f"Test total loss: {test_loss[2]:.5f}")

