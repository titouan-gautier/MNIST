import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Couche d'entrée du modèle = des images de taille 28,28 avec notre vecteur de gris 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.dropout2 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 100)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


def get_loss_fn():
    loss = nn.CrossEntropyLoss()
    loss.__name__ = 'CrossEntropyLoss'
    return loss


def get_model():
    model = Model()
    return model


def get_accuracy_metrics():
    class CustomAccuracy(nn.Module):
        __name__ = "accuracy"

        def __init__(self):
            super().__init__()

        def forward(self, y_pred, y_true):
            y_pred = torch.argmax(y_pred, dim=1)
            correct = (y_pred == y_true).float()
            accuracy = correct.sum() / len(correct)
            return accuracy

    metrics = [CustomAccuracy()]
    return metrics


def get_optimizer_fn(model,lr):
    return optim.Adam([dict(params=model.parameters(), lr=lr)])


def load_model(model_name):
    model = get_model()
    model.load_state_dict(torch.load('./models/' + model_name))

    return model
