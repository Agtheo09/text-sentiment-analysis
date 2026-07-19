from torch import nn
from torch.nn import functional as F
import torch

input_size = 368
output_size = 2
hidden_size = 500


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, X):
        X = torch.relu((self.fc1(X)))
        X = torch.relu((self.fc2(X)))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)


# class Net(nn.Module):
#     def __init__(self, input_size=320, num_classes=2, num_of_channels=1):
#         super(Net, self).__init__()

#         self.conv1 = nn.Conv1d(
#             in_channels=num_of_channels, out_channels=640, kernel_size=1
#         )
#         self.conv2 = nn.Conv1d(in_channels=640, out_channels=320, kernel_size=1)

#         conv_output_size = self.calculate_conv_output_size(input_size, num_of_channels)

#         self.fc1 = nn.Linear(conv_output_size, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 16)
#         self.fc4 = nn.Linear(16, num_classes)

#     def forward(self, x):
#         # Convolutional layers
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))

#         # Flattening
#         x = x.view(x.size(0), -1)

#         # Fully connected layers
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)

#         return x

#     def calculate_conv_output_size(self, input_size, num_channels):
#         dummy_input = torch.randn(1, num_channels)
#         conv1_output = F.relu(self.conv1(dummy_input))
#         conv2_output = F.relu(self.conv2(conv1_output))
#         return conv2_output.view(conv2_output.size(0), -1).size(1)
