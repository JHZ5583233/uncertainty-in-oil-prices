import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

level_converter: dict[int, str] = {0: "Low", 1: "Medium", 2: "High"}
country_converter: dict[int, str] = dict()


def split_input(df: pd.DataFrame) -> torch.Tensor:
    countries = df["country"].unique()
    country_indexer: dict[str, int] = dict()
    level_indexer: dict[str, int] = {"Low": 0, "Medium": 1, "High": 2}

    for index, country in enumerate(countries):
        country_converter[index] = country
        country_indexer[country] = index

    country_index: list[int] = []
    income_index: list[int] = []
    subsidy_index: list[int] = []

    for i, _ in enumerate(df["country"]):
        country_index.append(country_indexer[df["country"].iloc[i]])
        income_index.append(level_indexer[df["income_level"].iloc[i]])
        subsidy_index.append(level_indexer[df["subsidy_level"].iloc[i]])

    df["country_index"] = country_index
    df["income_index"] = income_index
    df["subsidy_index"] = subsidy_index

    return torch.Tensor(df["country_index", "income_index", "subsidy_index"])


def split_groundtruth(df: pd.DataFrame) -> torch.Tensor:
    truth_frame = df["brent_crude_usd"]

    return torch.Tensor(truth_frame)


if __name__ == "__main__":
    # Test split_groundtruth function
    import numpy as np

    # Create a sample dataframe
    test_df = pd.DataFrame(
        {
            "brent_crude_usd": [45.5, 60.2, 55.8, 70.1, 48.9],
            "other_column": ["a", "b", "c", "d", "e"],
        }
    )

    print("Test DataFrame:")
    print(test_df)
    print()

    # Test split_groundtruth
    result = split_groundtruth(test_df)

    print("Result type:", type(result))
    print("Result shape:", result.shape)
    print("Result dtype:", result.dtype)
    print("Result values:")
    print(result)
    print()

    # Verify values match
    expected = torch.Tensor(test_df["brent_crude_usd"].values)
    assert torch.allclose(result, expected), "Values don't match!"
    print("✓ Values match expected output")
    print()

    # Check for NaN or Inf
    assert not torch.isnan(result).any(), "Result contains NaN values!"
    assert not torch.isinf(result).any(), "Result contains Inf values!"
    print("✓ No NaN or Inf values")
