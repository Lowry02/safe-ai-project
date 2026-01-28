import torch
import torch.nn as nn
    
class EncoderWithPooling(nn.Module):
    def __init__(self, in_channels:int = 3, proj_dim:int = 128):
        super(EncoderWithPooling, self).__init__()

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
            nn.Linear(256 * 4 * 4, proj_dim * 4),
            nn.ReLU(),
            nn.Linear(proj_dim * 4, proj_dim * 2),
            nn.ReLU(),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.BatchNorm1d(proj_dim)
        )
    
    def forward(self, x:torch.Tensor):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.projector(x)
        return x
    
class EncoderNoPooling(nn.Module):
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
    def __init__(self, in_channels:int = 3, num_classes:int = 10, proj_dim:int = 128, pooling:bool=True):
        super().__init__()

        if pooling:
            self.encoder = EncoderWithPooling(in_channels, proj_dim)
        else: 
            self.encoder = EncoderNoPooling(in_channels, proj_dim)
            
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
    # it must return only logits in order to be verifiable by ABCrown
    def __init__(self, in_channels=3, num_classes=10, proj_dim=128, pooling:bool=True):
        super().__init__(in_channels, num_classes, proj_dim, pooling)
        
    def forward(self, x):
        return super().forward(x)[1]