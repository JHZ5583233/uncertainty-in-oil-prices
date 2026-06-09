import pandas
from pathlib import Path


def dataformatter(csv_path: Path) -> pandas.DataFrame:
    if not csv_path.exists():
        raise ValueError("Data path does not exist.")
    elif csv_path.suffix != ".csv":
        raise ValueError("Is not a csv file.")

    return pandas.read_csv(csv_path, header=1)

