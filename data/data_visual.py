from pathlib import Path
from pandas import DataFrame

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_tools import dataformatter


def visual_data_line(
    dataframe: DataFrame, 
    title: str,
    data_collumn: str,
    countries: list[str] = [],
    ) -> None:

    if countries == []:
        countries = dataframe["country"].unique()

    dataframe["date"] = pd.to_datetime(dataframe["date"])

    plt.figure(figsize=(14, 8))

    for country in countries:
        country_data = dataframe[dataframe["country"] == country].sort_values("date")
        plt.plot(
            country_data["date"],
            country_data[data_collumn],
            label=country,
            marker="o",
            linewidth=2,
            markersize=4,
            alpha=0.7,
        )

    plt.xlabel("Date", fontsize=12, fontweight="bold")
    plt.ylabel(data_collumn, fontsize=12, fontweight="bold")
    plt.title(
        title, fontsize=14, fontweight="bold"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()


def visual_data_table(
    dataframe: DataFrame, 
    title: str,
    data_collumn: list[str] = [],
    countries: list[str] = [],
    ) -> None:
    plt.table()



if __name__ == "__main__":

    data = dataformatter(Path("./global_fuel_prices_2020_2026.csv"))
    
    visual_data_line(
        data, 
        "yes",
        "tax_percentage",
    )

    visual_data_table(
        data,
        "income and subsidy level",
        ["income_level", "subsidy_level"]
        
    )

    plt.show()
