import torch
import torch.nn as nn


class Neg_Pearson_Loss(nn.Module):
    def __init__(self):
        super(Neg_Pearson_Loss, self).__init__()

    def forward(self, X, Y):
        # Check for NaNs
        assert not torch.any(torch.isnan(X)), "X contains NaNs"
        assert not torch.any(torch.isnan(Y)), "Y contains NaNs"

        # Normalize: Subtract mean
        X = X - X.mean(dim=1, keepdim=True)
        Y = Y - Y.mean(dim=1, keepdim=True)

        # Standardize: Divide by standard deviation
        X = X / (X.std(dim=1, keepdim=True) + 1e-5)
        Y = Y / (Y.std(dim=1, keepdim=True) + 1e-5)

        # Compute Pearson correlation
        correlation = torch.sum(X * Y, dim=1) / X.size(1)

        # Compute loss: 1 - mean correlation
        loss = 1 - correlation.mean()
        return loss
