import torch
import torch.nn as nn

class CNN1DClassifier(nn.Module):
    def __init__(self, input_length=95, num_classes=2):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.pool3 = nn.AdaptiveMaxPool1d(16)
        self.flattened_size = 64 * 16

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, L]
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)