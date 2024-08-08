import torch

from torch import nn
from typing import Callable


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    "A function that calculates the accuracy of a batch of predictions against the targets"
    cnt = torch.eq(torch.eq(predictions, targets).sum(dim=1), predictions.shape[1]).sum()
    return cnt.item() / len(predictions)


def max_to_one_hot(tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor of logits into a one-hot encoded tensor.

    This function takes a 2D tensor where each row represents a set of logits for different classes,
    and converts it into a one-hot encoded tensor where the maximum value in each row is set to 1, and all other values
    are set to 0.
    """
    _, max_indices = tensor.max(dim=1)
    one_hot_tensor = torch.zeros_like(tensor)
    one_hot_tensor.scatter_(1, max_indices.unsqueeze(1), 1)
    return one_hot_tensor


def train_one_epoch(model: nn.Module, 
                    loader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim, 
                    grad_accum_size: int, 
                    acc_fn: Callable[[torch.Tensor, torch.Tensor], float]):
    """
    A function that trains a model by going through all the mini-batches in the training dataloader once.
    """
    print("\tTraining...")
    model.train()
    accum_loss = 0  # accumulation of the loss of each batch
    accum_acc = 0  # accumulation of the accuracy of each batch
    
    for i, (X, y) in enumerate(loader):
        if i == 5072:
                outputs = model(input_ids=X["input_ids"], attention_mask=X["attention_mask"], token_type_ids=X["token_type_ids"],
                                pixel_values=((torch.rand(7, 6, 3, 352, 608) * 2) - 1).to("cuda"), pixel_mask=torch.ones(7, 6, 352, 608).to("cuda"), labels=y)
        else:
            outputs = model(input_ids=X["input_ids"], attention_mask=X["attention_mask"], token_type_ids=X["token_type_ids"],
                            pixel_values=((torch.rand(8, 6, 3, 352, 608) * 2) - 1).to("cuda"), pixel_mask=torch.ones(8, 6, 352, 608).to("cuda"), labels=y)
        loss = outputs.loss
        loss /= grad_accum_size
        loss.backward()
        pred = max_to_one_hot(outputs.logits)
        acc = acc_fn(pred, y)

        accum_loss += loss.item()
        accum_acc += acc

        if (i + 1) % grad_accum_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        print(f"\t\tBatch {i+1}/{len(loader)}: Loss {loss} | Accuracy: {acc*100}%")

    avg_loss = accum_loss / len(loader)
    avg_acc = accum_acc / len(loader)

    return avg_loss, avg_acc


def val_step(model: nn.Module, 
             loader: torch.utils.data.DataLoader, 
             acc_fn: Callable[[torch.Tensor, torch.Tensor], float]):
    """
    A function that validates a model by going through all the mini-batches in the validation dataloader once.
    """
    print("\tValidating...")
    model.eval()
    accum_loss = 0  # accumulation of the loss of each batch
    accum_acc = 0  # accumulation of the accuracy of each batch

    with torch.inference_mode():
        for i, (X, y) in enumerate(loader):
            if i == 319:
                outputs = model(input_ids=X["input_ids"], attention_mask=X["attention_mask"], token_type_ids=X["token_type_ids"],
                                pixel_values=((torch.rand(4, 6, 3, 352, 608) * 2) - 1).to("cuda"), pixel_mask=torch.ones(4, 6, 352, 608).to("cuda"), labels=y)
            else:
                outputs = model(input_ids=X["input_ids"], attention_mask=X["attention_mask"], token_type_ids=X["token_type_ids"],
                                pixel_values=((torch.rand(8, 6, 3, 352, 608) * 2) - 1).to("cuda"), pixel_mask=torch.ones(8, 6, 352, 608).to("cuda"), labels=y)
            loss = outputs.loss
            pred = max_to_one_hot(outputs.logits)
            acc = acc_fn(pred, y)

            accum_loss += loss.item()
            accum_acc += acc

    avg_loss = accum_loss / len(loader)
    avg_acc = accum_acc / len(loader)
    
    return avg_loss, avg_acc


def trainjob(model: nn.Module, 
             epochs: int, 
             train_loader: torch.utils.data.DataLoader, 
             val_loader: torch.utils.data.DataLoader, 
             optimizer: torch.optim, 
             scheduler: torch.optim.lr_scheduler, 
             grad_accum_size: int, 
             acc_fn: Callable[[torch.Tensor, torch.Tensor], float] = accuracy):
    """
    A function to train a model for a specific number of epochs.
    """
    # 4 lists to save the results at the end of each epoch
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        if scheduler is not None:
            print(f"\tLearning rate: {scheduler.get_last_lr()[0]}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, grad_accum_size, acc_fn)
        print(f"\tTrain Loss: {train_loss} | Train Accuracy: {train_acc*100}%")
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = val_step(model, val_loader, acc_fn)
        print(f"\tValidation Loss: {val_loss} | Validation Accuracy: {val_acc*100}%")
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if scheduler is not None:
            scheduler.step()

    return train_losses, train_accs, val_losses, val_accs
