import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Callable, Tuple
from torch.utils.data import DataLoader


def train(
    model:nn.Module,
    train_loader:DataLoader,
    validation_loader:DataLoader,
    optimizer:Optimizer,
    criterion:nn.Module,
    epochs:int,
    device:torch.device|str,
    middleware:Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None
) -> nn.Module:
    
    def default_middleware(images:torch.Tensor, labels:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return images, labels
    
    if middleware is None:
        middleware = default_middleware
        
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for _, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            images, labels = middleware(images, labels)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            validation_loss = 0
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                validation_loss += loss.item()

        validation_loss /= len(validation_loader)

        print(f"> Epoch {epoch+1}/{epoch}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {validation_loss:.4f}")
        
    return model

