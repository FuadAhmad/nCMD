import torch
from torch import nn
from torch.nn import functional as F


# MLP for Binary Classification
class MLP_Binary(nn.Module):
    def __init__(self, input_shape, num_units=64):
        super(MLP_Binary, self).__init__()
        self.fc1 = nn.Linear(input_shape, num_units) #-, 40)
        self.fc2 = nn.Linear(num_units, 32)
        self.fc3 = nn.Linear(32, 1)
        # define dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x) #- un commented
        x = torch.relu(self.fc2(x))
        x = self.dropout(x) #- un commented
        x = torch.sigmoid(self.fc3(x))
        return x


# MLP for Multi-class Classification
class MLP_Mult(nn.Module):
    def __init__(self, input_shape, num_units=64, num_classes=6):
        super(MLP_Mult, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, num_units)
        self.fc4 = nn.Linear(num_units, num_classes)
        # define dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.fc4(x)
        return x

# MLP for Multi-class Classification -----------hidden 1 layer nCMD
class MLP_Mult_hCMD(nn.Module):
    def __init__(self, input_shape, num_units=64, num_classes=6):
        super(MLP_Mult_hCMD, self).__init__()
        self.fc1 = nn.Linear(input_shape, num_units)
        self.fc4 = nn.Linear(num_units, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc4(x)
        return x
