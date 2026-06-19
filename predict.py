import torch

def pred_bnn(
    model,
    input_data: torch.Tensor,
) -> list[tuple[float, float]]:
    return_data:list[tuple[float, float]] = []

    model.eval()
    with torch.no_grad():
        for x in input_data:
            mean, variance = model(x)

            return_data.append((mean, variance))

    return return_data


if __name__ == "__main__":
    from pathlib import Path
    from pandas import read_csv

    from model.base_model import BayesianNeuralNetwork as BNN
    from data.data_loader import split_input
    
    model_path = Path("./save_model/m.pyt")
    
    model = BNN(2)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    dataframe = read_csv(Path("./data/global_fuel_prices_2020_2026.csv"))

    x = split_input(dataframe, ["Jordan"])

    print(x)

    preds = pred_bnn(model, torch.from_numpy(x.to_numpy()).float())

    print(preds)