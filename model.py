import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels:int = 3, proj_dim:int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=proj_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(proj_dim*4*4, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        return x

class LinearClassifier(nn.Module):
    def __init__(self, in_dim:int = 128, num_classes:int = 10):
        super(LinearClassifier, self).__init__()

        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x:torch.Tensor):
        return self.classifier(x)

class CNN(nn.Module):
    def __init__(self, in_channels:int = 3, num_classes:int = 10, proj_dim:int = 128):
        super().__init__()

        self.encoder = Encoder(in_channels, proj_dim)
        self.classifier = LinearClassifier(proj_dim, num_classes)

    @classmethod
    def import_from(cls, encoder:nn.Module, classifier:nn.Module):
        cnn = CNN()
        cnn.encoder = encoder
        cnn.classifier = classifier

        return cnn

    def forward(self, x:torch.Tensor):
        embeddings = self.encoder(x)
        logits = self.classifier(embeddings)
        return embeddings, logits

class CNNCrown(CNN):
    # same as CNN but it returns only logits (useful during verification with abcrown)
    def __init__(self, in_channels=3, num_classes=10, proj_dim=128):
        super().__init__(in_channels, num_classes, proj_dim)

    def forward(self, x):
        return super().forward(x)[1]