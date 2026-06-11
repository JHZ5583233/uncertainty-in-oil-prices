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
    countries: list[str] = [],
    ) -> None:
    if countries == []:
        countries = dataframe["country"].unique()

    income_levels = sorted(dataframe["income_level"].unique())
    subsidy_levels = sorted(dataframe["subsidy_level"].unique())

    columns = []
    for income in income_levels:
        for subsidy in subsidy_levels:
            columns.append(f"{income}/{subsidy}")

    table_data = []
    for country in countries:
        row = [country]
        country_data = dataframe[dataframe["country"] == country]

        for income in income_levels:
            for subsidy in subsidy_levels:
                # Count occurrences of this combination
                count = len(
                    country_data[
                        (country_data["income_level"] == income)
                        & (country_data["subsidy_level"] == subsidy)
                    ]
                )
                row.append(count)

        table_data.append(row)

    fig, ax = plt.subplots(figsize=(12, len(countries) * 0.5 + 1))
    ax.axis("off")

    full_table_data = [["Country"] + columns] + table_data

    table = ax.table(
        cellText=full_table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.15] + [0.12] * len(columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(len(full_table_data[0])):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(1, len(full_table_data)):
        color = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(len(full_table_data[0])):
            table[(i, j)].set_facecolor(color)

    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()



if __name__ == "__main__":

    data = dataformatter(Path("./global_fuel_prices_2020_2026.csv"))
    
    # visual_data_line(
    #     data, 
    #     "yes",
    #     "tax_percentage",
    # )

    visual_data_table(
        data,
        "income and subsidy level",
    )

    plt.show()
