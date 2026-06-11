from model.base_model import BayesianNeuralNetwork
import torch
from torch import optim


def train_bnn(
    model: BayesianNeuralNetwork, 
    train_loader, 
    test_loader, 
    optimizer,
    epochs=50, 
    beta=1.0
):
    """Train a Bayesian neural network using BCE loss + KL regularization.

    Args:
        model: Bayesian neural network to train.
        train_loader: Training data loader yielding feature and label batches.
        test_loader: Evaluation data loader yielding feature and label batches.
        optimizer: Optimizer configured with model parameters.
        epochs: Number of training epochs.
        beta: KL divergence scaling factor.
    """
    dataset_size = len(train_loader.dataset)
    device = model.device

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).view(-1, 1)

            # Forward pass
            with torch.set_grad_enabled(True):
                preds = model(x_batch)
                kl = model.kl_div()
                nll = torch.nn.functional.binary_cross_entropy(preds, y_batch)
                loss = nll + beta * kl / dataset_size

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).view(-1, 1)
                preds = model(x_batch)
                predicted = (preds > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        acc = correct / total
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Test Acc: {acc:.4f}")               
            