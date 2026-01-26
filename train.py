# TRAIN FILE
import wandb
import torch
import torch.nn as nn
from typing import Callable
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

def default_middleware(images:torch.Tensor, labels:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return images, labels

def train(
    model:nn.Module,
    train_loader:DataLoader,
    validation_loader:DataLoader,
    optimizer:Optimizer,
    criterion:nn.Module,
    epochs:int,
    device:torch.device|str,
    scheduler: LRScheduler | None = None,
    middleware:Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
    compute_accuracy:bool = True,
    wandb_logging: bool = False,
) -> nn.Module:
    if middleware is None:
        middleware = default_middleware
        
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0
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
            train_accuracy += (torch.argmax(output, dim=1) == labels).sum().item() / len(labels)

        if scheduler is not None:
            scheduler.step()
            
        train_loss /= len(train_loader)
        train_accuracy = train_accuracy / len(train_loader) * 100
        
        model.eval()
        with torch.no_grad():
            validation_loss = 0
            validation_accuracy = 0
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                images, labels = middleware(images, labels)
                
                output = model(images)
                loss = criterion(output, labels)
                
                validation_loss += loss.item()
                validation_accuracy += (torch.argmax(output, dim=1) == labels).sum().item() / len(labels)

        validation_loss /= len(validation_loader)
        validation_accuracy = validation_accuracy / len(validation_loader) * 100
        
        if compute_accuracy:
            print(f"> Epoch {epoch+1}/{epochs}")
            print(f"  Training loss      : {train_loss:.4f}, Training accuracy  : {train_accuracy:.2f}%")
            print(f"  Validation loss    : {validation_loss:.4f}, Validation accuracy: {validation_accuracy:.2f}%")
        else:
            print(f"> Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Validation Loss: {validation_loss:.4f}")
            
        if wandb_logging:
            log = {
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
            
            if compute_accuracy:
                log["train_accuracy"] = train_accuracy
                log["validation_accuracy"] = validation_accuracy
            
            wandb.log(log)
            
        
    return model

