import torch
import torch.nn as nn

class _CNN(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 10, proj_dim = 128):
        super(_CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Linear(256 * 4 * 4, 1024)

        self.projector = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, proj_dim)
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        features = self.fc(x)

        proj = self.projector(features)      # for SupCon
        logits = self.classifier(features)   # for CE

        return proj, logits
    
class CNN(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 10, proj_dim = 128):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.projector = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, proj_dim),
            nn.BatchNorm1d(proj_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim*4),
            nn.ReLU(),
            nn.Linear(proj_dim*4, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        
        proj = self.projector(x)      # for SupCon
        logits = self.classifier(proj)   # for CE

        return proj, logits
    
    
class CNNCrown(CNN):
    # it must return only logits in order to be verifiable by ABCrown
    def __init__(self, in_channels=3, num_classes=10, proj_dim=128):
        super().__init__(in_channels, num_classes, proj_dim)
        
    def forward(self, x):
        return super().forward(x)[1]