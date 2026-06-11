import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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
    input_data: pd.DataFrame, output_data: pd.DataFrame, test_size: float = 0.2
) -> tuple[DataLoader, DataLoader]:
    X_train, X_test, Y_train, Y_test = train_test_split(
        input_data, output_data, test_size=test_size
    )
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
            "tax_percentage": [5.5, 10.2, 8.7, 6.1, 9.3],
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
        print("Result dtypes:")
        print(result_input.dtypes)
        print("Result values:")
        print(result_input)
        print()

        # Verify shape
        assert result_input.shape[0] == len(test_df_input), "Row count mismatch!"
        assert result_input.shape[1] == 4, (
            "Should have 4 columns (country, income, subsidy, tax)!"
        )
        print("✓ Shape is correct (5 rows, 4 columns)")

        # Verify value ranges for numeric columns
        numeric_cols = [
            "country_index",
            "income_index",
            "subsidy_index",
            "tax_percentage",
        ]
        for col in numeric_cols:
            assert (result_input[col] >= 0).all(), f"{col} should be non-negative!"
        print("✓ All values are non-negative")

        # Check country_converter was populated
        assert len(country_converter) > 0, "country_converter should be populated!"
        print(f"✓ country_converter populated: {country_converter}")

        # Verify indices are in expected ranges
        assert (result_input["income_index"] >= 0).all() and (
            result_input["income_index"] <= 2
        ).all(), "Income indices out of range!"
        assert (result_input["subsidy_index"] >= 0).all() and (
            result_input["subsidy_index"] <= 2
        ).all(), "Subsidy indices out of range!"
        print("✓ Income and subsidy indices in valid range [0-2]")

        # Verify we can convert to tensor
        tensor_result = torch.tensor(result_input.values)
        assert tensor_result.shape == (5, 4), "Tensor shape mismatch!"
        print("✓ Result can be converted to PyTorch tensor")

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
        print("Result dtypes:")
        print(result_truth.dtypes)
        print("Result values:")
        print(result_truth)
        print()

        # Verify values match
        expected_values = test_df_truth["brent_crude_usd"].values
        result_values = result_truth["brent_cruse_usd"].values
        assert np.allclose(result_values, expected_values), "Values don't match!"
        print("✓ Values match expected output")
        print()

        # Check for NaN or Inf
        assert not result_truth.isna().any().any(), "Result contains NaN values!"
        assert not np.isinf(result_truth.values).any(), "Result contains Inf values!"
        print("✓ No NaN or Inf values")
        print()

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
    print("Testing data_to_loader function")
    print("=" * 60)

    # Create sample dataframes for data_to_loader
    test_input_data = pd.DataFrame(
        {
            "country_index": [0, 1, 2, 0, 1],
            "income_index": [0, 1, 2, 1, 0],
            "subsidy_index": [1, 2, 0, 1, 2],
            "tax_percentage": [5.5, 10.2, 8.7, 6.1, 9.3],
        }
    ).astype(float)

    test_output_data = pd.DataFrame(
        {
            "brent_cruse_usd": [45.5, 60.2, 55.8, 70.1, 48.9],
        }
    ).astype(float)

    print("\nInput DataFrame:")
    print(test_input_data)
    print("\nOutput DataFrame:")
    print(test_output_data)
    print()

    # Test data_to_loader
    try:
        train_loader, test_loader = data_to_loader(
            test_input_data.values, test_output_data.values, test_size=0.4
        )
        print("✓ data_to_loader executed successfully")
        print(f"Train loader type: {type(train_loader)}")
        print(f"Test loader type: {type(test_loader)}")
        print()

        # Verify loaders are DataLoader instances
        assert isinstance(train_loader, DataLoader), (
            "train_loader should be a DataLoader!"
        )
        assert isinstance(test_loader, DataLoader), (
            "test_loader should be a DataLoader!"
        )
        print("✓ Both outputs are DataLoader instances")
        print()

        # Check batch sizes
        train_batches = list(train_loader)
        test_batches = list(test_loader)
        print(f"Number of training batches: {len(train_batches)}")
        print(f"Number of test batches: {len(test_batches)}")
        print()

        # Verify batch structure
        assert len(train_batches) > 0, "Train loader should have batches!"
        assert len(test_batches) > 0, "Test loader should have batches!"
        print("✓ Both loaders contain batches")
        print()

        # Check data shapes in batches
        first_train_batch = train_batches[0]
        X_batch, Y_batch = first_train_batch
        print(f"Training batch X shape: {X_batch.shape}")
        print(f"Training batch Y shape: {Y_batch.shape}")
        print()

        # Verify tensor types
        assert isinstance(X_batch, torch.Tensor), "X batch should be a Tensor!"
        assert isinstance(Y_batch, torch.Tensor), "Y batch should be a Tensor!"
        print("✓ Batches contain PyTorch tensors")
        print()

        # Verify train/test split ratio
        total_samples = len(test_input_data)
        expected_test_size = int(total_samples * 0.4)
        actual_test_size = sum(len(batch[0]) for batch in test_batches)
        actual_train_size = sum(len(batch[0]) for batch in train_batches)
        print(f"Total samples: {total_samples}")
        print(f"Expected test size (40%): ~{expected_test_size}")
        print(f"Actual test size: {actual_test_size}")
        print(f"Actual train size: {actual_train_size}")
        print()

        # Verify no data overlap between train and test
        assert actual_train_size + actual_test_size == total_samples, (
            "Train and test sets should cover all samples!"
        )
        print("✓ No data loss in train/test split")
        print()

        # Test with default test_size
        train_loader_default, test_loader_default = data_to_loader(
            test_input_data.values, test_output_data.values
        )
        test_batches_default = list(test_loader_default)
        train_batches_default = list(train_loader_default)
        default_test_size = sum(len(batch[0]) for batch in test_batches_default)
        default_train_size = sum(len(batch[0]) for batch in train_batches_default)
        print(f"Default test size (20%): {default_test_size}")
        print(f"Default train size: {default_train_size}")
        print("✓ Default test_size=0.2 works correctly")

    except Exception as e:
        print(f"✗ Error in data_to_loader: {e}")
        import traceback

        traceback.print_exc()

    print()
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
