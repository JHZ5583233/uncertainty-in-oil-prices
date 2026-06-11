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
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).view(-1, 1).float()

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
                x_batch = x_batch.to(device).float()
                y_batch = y_batch.to(device).view(-1, 1).float()
                mean, varience = model(x_batch)

                error = (mean - y_batch)**2
                sum_error += sum(error)
                

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | cumulative error {sum_error}")


if __name__ == "__main__":
    from pathlib import Path
    from pandas import read_csv
    
    from model.base_model import BayesianNeuralNetwork as BNN
    from data.data_loader import split_input, split_groundtruth, data_to_loader

    model = BNN(3)

    dataframe = read_csv(Path("./data/global_fuel_prices_2020_2026.csv"))

    input_b = split_input(dataframe)
    output_b = split_groundtruth(dataframe)

    train_l, test_l = data_to_loader(input_b, output_b)

    train_bnn(
        model, 
        train_l, 
        test_l, 
        torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    )
