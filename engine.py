import torch


def accuracy(predictions: torch.Tensor, targets: torch.Tensor, answers_len: int) -> float:
    cnt = torch.eq(torch.eq(predictions, targets).sum(dim=1), answers_len).sum()
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


def train_one_epoch(model, loader, optimizer, acc_fn, answ_len):
    """
    A function that trains the model by going through all the mini-batches in the training dataloader once.
    """
    print("\tTraining...")
    model.train()
    losses = []  # to save the loss of each mini-batch in order to take their average at the end
    accuracies = []  # to save the accuracy of each mini-batch in order to take their average at the end

    for i, (X, y) in enumerate(loader):
        outputs = model(**X, labels=y)
        loss = outputs.loss
        pred = max_to_one_hot(outputs.logits)
        acc = acc_fn(pred, y, answ_len)

        losses.append(loss.item())
        accuracies.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"\t\tBatch {i+1}/{len(loader)}: Loss {loss} | Accuracy: {acc*100}%")

    avg_loss = sum(losses) / len(loader)
    avg_acc = sum(accuracies) / len(loader)

    return avg_loss, avg_acc


def val_step(model, loader, acc_fn, answ_len):
    """
    A function that validates the model by going through all the mini-batches in the validation dataloader once.
    """
    print("\tValidating...")
    model.eval()
    losses = []  # to save the loss of each mini-batch in order to take their average at the end
    accuracies = []  # to save the accuracy of each mini-batch in order to take their average at the end

    with torch.inference_mode():
        for i, (X, y) in enumerate(loader):
            outputs = model(**X, labels=y)
            loss = outputs.loss
            pred = max_to_one_hot(outputs.logits)
            acc = acc_fn(pred, y, answ_len)

            losses.append(loss.item())
            accuracies.append(acc)

    avg_loss = sum(losses) / len(loader)
    avg_acc = sum(accuracies) / len(loader)
    
    return avg_loss, avg_acc


def trainjob(model, epochs, train_loader, val_loader, optimizer, answ_len, acc_fn=accuracy):
    """
    A function to train the model for a specific number of epochs.
    """
    # 4 lists to save the results at the end of each epoch
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, acc_fn, answ_len)
        print(f"\tTrain Loss: {train_loss} | Train Accuracy: {train_acc*100}%")
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = val_step(model, val_loader, acc_fn, answ_len)
        print(f"\tValidation Loss: {val_loss} | Validation Accuracy: {val_acc*100}%")
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    return train_losses, train_accs, val_losses, val_accs
