import torch

def relative_error(y_pred, y_test):
    errors = []
    for pred, actual in zip(y_pred, y_test):
        if actual != 0:
            error = abs(pred - actual) / abs(actual)
            errors.append(error)
        else:
            # Handle the case where the actual value is zero to avoid division by zero
            errors.append(float('inf'))  # Assigning infinity for the relative error
    return errors


def median_relative_error(predictions, targets):
    absolute_errors = torch.abs(predictions - targets)
    relative_errors = absolute_errors / (torch.abs(targets) + 1e-8)
    return torch.median(relative_errors).item()