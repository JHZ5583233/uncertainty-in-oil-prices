from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_tools import dataformatter


def visual_data_whole(
    path: Path, 
    data: str, 
    title: str) -> None:

    df = dataformatter(path)

    df["date"] = pd.to_datetime(df["date"])

    plt.figure(figsize=(14, 8))

    for country in df["country"].unique():
        country_data = df[df["country"] == country].sort_values("date")
        plt.plot(
            country_data["date"],
            country_data[data],
            label=country,
            marker="o",
            linewidth=2,
            markersize=4,
            alpha=0.7,
        )

    plt.xlabel("Date", fontsize=12, fontweight="bold")
    plt.ylabel(data, fontsize=12, fontweight="bold")
    plt.title(
        title, fontsize=14, fontweight="bold"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()



if __name__ == "__main__":
    visual_data_whole(
        Path("./global_fuel_prices_2020_2026.csv"), 
        data="tax_percentage",
        title="yes")

    plt.show()
