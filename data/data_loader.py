import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

level_converter: dict[int, str] = {0: "Low", 1: "Medium", 2: "High"}
country_converter: dict[int, str] = dict()


def split_input(df: pd.DataFrame) -> pd.DataFrame:
    r_dataframe = pd.DataFrame()
    
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

    r_dataframe["country_index"] = country_index
    r_dataframe["income_index"] = income_index
    r_dataframe["subsidy_index"] = subsidy_index
    r_dataframe["tax_percentage"] = df["tax_percentage"]

    return r_dataframe


def split_groundtruth(df: pd.DataFrame) -> pd.DataFrame:
    r_dataframe = pd.DataFrame()
    r_dataframe["brent_cruse_usd"] = df["brent_crude_usd"]

    return r_dataframe

def data_to_loader(
    input_data: pd.DataFrame, 
    output_data: pd.DataFrame,
    test_size: float =0.2
) -> tuple[DataLoader, DataLoader]:
    X_train, X_test, Y_train, Y_test = train_test_split(input_data, output_data, test_size=test_size)
    X_train_tensor = torch.tensor(X_train)
    Y_train_tensor = torch.tensor(Y_train)
    X_test_tensor = torch.tensor(X_test)
    Y_test_tensor = torch.tensor(Y_test)
    
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return (train_loader, test_loader)


if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("Testing split_input function")
    print("=" * 60)

    # Create a sample dataframe for split_input
    test_df_input = pd.DataFrame(
        {
            "country": ["USA", "USA", "China", "China", "India"],
            "income_level": ["High", "High", "Medium", "Medium", "Low"],
            "subsidy_level": ["Low", "Medium", "High", "Medium", "Low"],
        }
    )

    print("\nTest DataFrame:")
    print(test_df_input)
    print()

    # Test split_input
    try:
        result_input = split_input(test_df_input)
        print("✓ split_input executed successfully")
        print("Result type:", type(result_input))
        print("Result shape:", result_input.shape)
        print("Result dtype:", result_input.dtype)
        print("Result values:")
        print(result_input)
        print()

        # Verify shape
        assert result_input.shape[0] == len(test_df_input), "Row count mismatch!"
        assert result_input.shape[1] == 3, (
            "Should have 3 columns (country, income, subsidy)!"
        )
        print("✓ Shape is correct (5 rows, 3 columns)")

        # Verify value ranges
        assert (result_input >= 0).all(), "All indices should be non-negative!"
        print("✓ All values are non-negative indices")

        # Check country_converter was populated
        assert len(country_converter) > 0, "country_converter should be populated!"
        print(f"✓ country_converter populated: {country_converter}")

        # Verify indices are in expected ranges
        assert (result_input[:, 1] >= 0).all() and (result_input[:, 1] <= 2).all(), (
            "Income indices out of range!"
        )
        assert (result_input[:, 2] >= 0).all() and (result_input[:, 2] <= 2).all(), (
            "Subsidy indices out of range!"
        )
        print("✓ Income and subsidy indices in valid range [0-2]")

    except Exception as e:
        print(f"✗ Error in split_input: {e}")
        import traceback

        traceback.print_exc()

    print()
    print("=" * 60)
    print("Testing split_groundtruth function")
    print("=" * 60)

    # Create a sample dataframe for split_groundtruth
    test_df_truth = pd.DataFrame(
        {
            "brent_crude_usd": [45.5, 60.2, 55.8, 70.1, 48.9],
            "other_column": ["a", "b", "c", "d", "e"],
        }
    )

    print("\nTest DataFrame:")
    print(test_df_truth)
    print()

    # Test split_groundtruth
    try:
        result_truth = split_groundtruth(test_df_truth)
        print("✓ split_groundtruth executed successfully")
        print("Result type:", type(result_truth))
        print("Result shape:", result_truth.shape)
        print("Result dtype:", result_truth.dtype)
        print("Result values:")
        print(result_truth)
        print()

        # Verify values match
        expected = torch.Tensor(test_df_truth["brent_crude_usd"].values).unsqueeze(1)
        assert torch.allclose(result_truth, expected), "Values don't match!"
        print("✓ Values match expected output")
        print()

        # Check for NaN or Inf
        assert not torch.isnan(result_truth).any(), "Result contains NaN values!"
        assert not torch.isinf(result_truth).any(), "Result contains Inf values!"
        print("✓ No NaN or Inf values")

        # Verify shape
        assert result_truth.shape[0] == len(test_df_truth), "Row count mismatch!"
        assert result_truth.shape[1] == 1, "Should have 1 column (reshaped as 2D)!"
        print("✓ Shape is correct (5 rows, 1 column)")

    except Exception as e:
        print(f"✗ Error in split_groundtruth: {e}")
        import traceback

        traceback.print_exc()

    print()
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
