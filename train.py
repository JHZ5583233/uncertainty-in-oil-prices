import torch
from torch import optim

from tools import detect_device

loss = torch.nn.GaussianNLLLoss()

def train_bnn(
    model, 
    train_loader, 
    test_loader, 
    optimizer: optim.Optimizer, 
    epochs=50, 
    beta=1.0,
):
    device = detect_device()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).view(-1, 1)

            # Forward pass
            with torch.set_grad_enabled(True):
                mean, varience = model(x_batch)
                kl = model.kl_div()
    
                loss_val = loss(mean, y_batch, varience) + (beta * kl)
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                train_loss += loss_val.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        sum_error = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).view(-1, 1)
                mean, varience = model(x_batch)

                error = (mean - y_batch)**2
                sum_error += sum(error)
                

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | cumulative error {sum_error}")
