from typing import Callable, Tuple
import matplotlib.pyplot as plt
import io
import umap
import wandb
import torch
import torch.nn as nn
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
    middleware:Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = default_middleware,
    compute_accuracy:bool = True,
    wandb_logging: bool = False,
) -> nn.Module:
    """
    Trains a PyTorch model using the provided training and validation data loaders.
    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        validation_loader (DataLoader): DataLoader for the validation dataset.
        optimizer (Optimizer): Optimizer used for updating model parameters.
        criterion (nn.Module): Loss function used to compute the training and validation loss.
        epochs (int): Number of epochs to train the model.
        device (torch.device | str): Device to use for training (e.g., 'cpu' or 'cuda').
        scheduler (LRScheduler | None, optional): Learning rate scheduler. Defaults to None.
        middleware (Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]], optional): 
            Function to preprocess input images and labels. Defaults to `default_middleware` -> no preprocess is made.
        compute_accuracy (bool, optional): Whether to compute and display accuracy during training. Defaults to True.
        wandb_logging (bool, optional): Whether to log training metrics to Weights & Biases. Defaults to False.
    Returns:
        nn.Module: The trained PyTorch model.
    """
    
    assert isinstance(model, nn.Module), "model must be an instance of nn.Module"
    assert isinstance(train_loader, DataLoader), "train_loader must be an instance of DataLoader"
    assert isinstance(validation_loader, DataLoader), "validation_loader must be an instance of DataLoader"
    assert isinstance(optimizer, Optimizer), "optimizer must be an instance of Optimizer"
    assert isinstance(criterion, nn.Module), "criterion must be an instance of nn.Module"
    assert isinstance(epochs, int) and epochs > 0, "epochs must be a positive integer"
    assert isinstance(device, (torch.device, str)), "device must be a torch.device or a string"
    assert scheduler is None or isinstance(scheduler, LRScheduler), "scheduler must be None or an instance of LRScheduler"
    assert callable(middleware), "middleware must be a callable function"
    assert isinstance(compute_accuracy, bool), "compute_accuracy must be a boolean"
    assert isinstance(wandb_logging, bool), "wandb_logging must be a boolean"

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

def test(model:nn.Module, train_loader:DataLoader, test_loader:DataLoader, device:str|torch.device="cuda") -> Tuple[float, float]:
    """
    Evaluate the performance of a PyTorch model on training and testing datasets.
    Args:
        model (nn.Module): The PyTorch model to be evaluated.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        device (str, optional): The device to run the evaluation on ("cuda" or "cpu"). Defaults to "cuda".
    Returns:
        Tuple[float, float]: A tuple containing:
            - train_accuracy (float): The accuracy of the model on the training dataset (percentage).
            - test_accuracy (float): The accuracy of the model on the testing dataset (percentage).
    """

    assert isinstance(model, nn.Module), "model must be an instance of nn.Module"
    assert isinstance(train_loader, DataLoader), "train_loader must be an instance of DataLoader"
    assert isinstance(test_loader, DataLoader), "test_loader must be an instance of DataLoader"
    assert isinstance(device, str) or isinstance(device, torch.device), "device must be a string or a torch.device"

    model.eval()
    model.to(device)
    train_accuracy = 0
    with torch.no_grad():
        for _, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)

            train_accuracy += (torch.argmax(logits, dim=1) == labels).sum().item() / len(labels)

        train_accuracy = train_accuracy / len(train_loader) * 100

    test_accuracy = 0
    with torch.no_grad():
        for _, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)

            test_accuracy += (torch.argmax(logits, dim=1) == labels).sum().item() / len(labels)

        test_accuracy = test_accuracy / len(test_loader) * 100

    return train_accuracy, test_accuracy

def get_embeddings_plot(
    model: nn.Module, 
    train_loader: DataLoader, 
    validation_loader: DataLoader, 
    device:str|torch.device = "cuda", 
    N_ITERATIONS: int = 4
) -> io.BytesIO:
    
    """
    Generates a UMAP visualization of embeddings for both training and validation datasets.
    This function computes embeddings using the provided model for a subset of the training 
    and validation datasets, reduces their dimensionality to 2D using UMAP, and plots the 
    results side-by-side. The resulting plot is saved to an in-memory buffer and returned.
    Args:
        model (nn.Module): The neural network model used to compute embeddings.
        train_loader (DataLoader): DataLoader for the training dataset.
        validation_loader (DataLoader): DataLoader for the validation dataset.
        device (str | torch.device, optional): The device to run the model on ("cuda" or "cpu"). Defaults to "cuda".
        N_ITERATIONS (int, optional): The maximum number of iterations (batches) to process 
            from each DataLoader. Defaults to 4.
    Returns:
        io.BytesIO: A buffer containing the generated plot in PNG format.
    """
    
    assert isinstance(model, nn.Module), "model must be an instance of nn.Module"
    assert isinstance(train_loader, DataLoader), "train_loader must be an instance of DataLoader"
    assert isinstance(validation_loader, DataLoader), "validation_loader must be an instance of DataLoader"
    assert isinstance(device, (str, torch.device)), "device must be a string or a torch.device"
    assert isinstance(N_ITERATIONS, int) and N_ITERATIONS > 0, "N_ITERATIONS must be a positive integer"
    
    model.eval()

    all_embeddings = []
    all_labels = []

    # train embeddings
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

            if i >= N_ITERATIONS:
                break

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    # umap computation
    umap_reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )

    embeddings_2d = umap_reducer.fit_transform(embeddings)

    # plotting
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=5
    )
    plt.title("UMAP of Embeddings (Train)")
    plt.colorbar(scatter, ticks=range(10))

    # ---- validatin embeddings
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(validation_loader):
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

            if i >= N_ITERATIONS:
                break

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    # umap computation
    umap_reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )

    embeddings_2d = umap_reducer.fit_transform(embeddings)

    plt.subplot(1, 2, 2)
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=5
    )
    plt.title("UMAP of Embeddings (Validation)")
    plt.colorbar(scatter, ticks=range(10))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

class RandomGaussianNoise(nn.Module):
    """
    RandomGaussianNoise is a PyTorch module that applies random Gaussian noise to input tensors 
    for data augmentation. This is particularly useful in training neural networks to improve 
    robustness and generalization by introducing slight variations to the input data.
    Attributes:
        mean (float): The mean of the Gaussian noise distribution. Default is 0.0.
        std (float): The standard deviation of the Gaussian noise distribution. Default is 0.05.
        p (float): The probability of applying the noise to the input. Default is 0.5.
    """
    
    def __init__(self, mean:float=0.0, std:float=0.05, p:float=0.5) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if torch.rand(1) > self.p:
            return x

        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise