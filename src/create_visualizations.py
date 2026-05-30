"""
Script to generate multi‑dimensional data visualizations for the
CMP_SC‑8630 data visualization assignment.  The script loads three
real‑world datasets related to climate and hydrology and produces
visualizations that explore patterns across multiple variables and
dimensions.  The resulting figures are saved to the ``output``
directory.  The datasets used here include:

* ``weather_data.csv`` – daily weather observations for multiple
  cities in New Zealand (2016–2017) containing temperature,
  humidity, wind, pressure and precipitation variables.  Source:
  mosaicData package within the Rdatasets collection.
* ``global_temp.csv`` – NASA Goddard Institute for Space Studies
  (GISTEMP) global land–ocean temperature anomalies from 1880 to
  2025.  Monthly anomalies relative to the 1951–1980 baseline are
  provided.  Source: NASA GISS via data.giss.nasa.gov.
* ``minnesota_weather.csv`` – monthly weather summary for six
  Minnesota agricultural sites (1927–1936) including cooling and
  heating degree days, precipitation and temperature extremes.
  Source: agridat package within Rdatasets.

The visualizations include heatmaps, scatter plots and line charts
to illustrate how variables such as temperature, humidity and
precipitation vary over time and across different locations.
"""

import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np


def ensure_output_dir(path: str) -> None:
    """Ensure that the output directory exists."""
    os.makedirs(path, exist_ok=True)


def plot_weather_heatmap(df: pd.DataFrame, outdir: str) -> str:
    # 1. & 2. Group by 'city' and 'month', mean of 'avg_temp', then pivot
    pivot_df = pd.pivot_table(df, values='avg_temp', index='city', columns='month', aggfunc='mean')
    
    # 3. Ensure the columns (months) are sorted in calendar order
    pivot_df = pivot_df.sort_index(axis=1)

    # 4. Create a heatmap using seaborn.heatmap()
    plt.figure(figsize=(10, 4))
    ax = sns.heatmap(pivot_df, cmap='coolwarm', annot=True, fmt='.1f',
                     cbar_kws={'label': 'Average temperature'})
    
    # 5. Set labels and title
    ax.set_title("Average monthly temperature by city")
    ax.set_xlabel("Month")
    ax.set_ylabel("City")

    # 6. Save the figure
    filepath = os.path.join(outdir, "weather_heatmap.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    return filepath
    # raise NotImplementedError("plot_weather_heatmap is not implemented yet.")


def plot_weather_scatter(df: pd.DataFrame, outdir: str) -> str:
    df_clean = df.copy()
    
    # 1. Clean the 'precip' column
    df_clean['precip'] = pd.to_numeric(df_clean['precip'], errors='coerce').fillna(0.0)

    # 2. Set up the figure
    plt.figure(figsize=(9, 6))
    
    # 3. Set a marker size range
    size_range = (20, 300)

    # 4. Generate a scatter plot
    ax = sns.scatterplot(
        data=df_clean,
        x="avg_humidity",
        y="avg_temp",
        hue="city",
        size="precip",
        sizes=size_range,
        alpha=0.65,
        legend=False
    )
    
    ax.set_xlabel("Average relative humidity (%)")
    ax.set_ylabel("Average temperature (°F)")
    
    # 5. Create a custom legend for the cities (Thêm markeredgecolor để có viền)
    unique_cities = df_clean['city'].unique()
    palette = sns.color_palette(n_colors=len(unique_cities))
    city_handles = [mlines.Line2D([], [], markerfacecolor=palette[i], markeredgecolor='dimgray', 
                                  marker='o', linestyle='None', markersize=8, label=city)
                    for i, city in enumerate(unique_cities)]
    
    leg1 = ax.legend(handles=city_handles, title="City", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(leg1)

    # 6. Create a custom legend for precipitation sizes (Cố định các mốc giống ảnh gốc)
    precip_vals = [0, 0.13, 0.26, 0.39]
    min_p, max_p = df_clean['precip'].min(), df_clean['precip'].max()
    
    mapped_sizes = np.interp(precip_vals, (min_p, max_p), size_range)
    
    precip_handles = [plt.scatter([], [], s=s_val, facecolors='darkgray', edgecolors='dimgray', 
                                  alpha=0.65, label=f"{p_val:g}")
                      for p_val, s_val in zip(precip_vals, mapped_sizes)]
    
    ax.legend(handles=precip_handles, title="Precipitation", loc="lower left", bbox_to_anchor=(1.02, 0.0), scatterpoints=1)

    # 7. Add title
    ax.set_title("Daily weather: temperature vs humidity with precipitation (size)")

    # 8. Save the figure
    filepath = os.path.join(outdir, "weather_scatter.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    return filepath
    # raise NotImplementedError("plot_weather_scatter is not implemented yet.")


def plot_global_temp_heatmap(df: pd.DataFrame, outdir: str) -> str:
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # 1. Reshape the dataframe from wide to long
    long_df = pd.melt(df, id_vars=['Year'], value_vars=months,
                      var_name='Month', value_name='Anomaly')

    # 2. Map Month abbreviations to month numbers
    month_map = {m: i+1 for i, m in enumerate(months)}
    long_df['MonthNum'] = long_df['Month'].map(month_map)

    # 3. Pivot back to a matrix
    pivot_df = long_df.pivot(index='Year', columns='MonthNum', values='Anomaly')
    
    # 4. Sort by year index
    pivot_df = pivot_df.sort_index(ascending=True)

    # 5. Set up the figure
    plt.figure(figsize=(10, 8))
    
    # 6. Draw a heatmap
    ax = sns.heatmap(pivot_df, cmap='coolwarm', vmin=-1.5, vmax=1.5,
                     linewidths=0, linecolor="white",
                     cbar_kws={'label': 'Temperature anomaly (°C relative to 1951–1980)'})

    # 7. Customize x-ticks
    ax.set_xticks(np.arange(12) + 0.5)
    ax.set_xticklabels(months, rotation=45)

    # 8. Set titles and labels
    ax.set_title("Global land–ocean temperature anomalies (1880–2025)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    # 9. Save the figure
    filepath = os.path.join(outdir, "global_temp_heatmap.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    return filepath
    # raise NotImplementedError("plot_global_temp_heatmap is not implemented yet.")


def plot_minnesota_precip_line(df: pd.DataFrame, outdir: str) -> str:
    df_clean = df.copy()
    
    # 1. Create a datetime column combining 'year' and 'mo'
    df_clean['date'] = pd.to_datetime({'year': df_clean['year'], 'month': df_clean['mo'], 'day': 1})

    # 2. Set up the figure
    plt.figure(figsize=(10, 6))

    # 3. Create a line plot
    ax = sns.lineplot(data=df_clean, x="date", y="precip", hue="site")

    # 4. Set title and labels
    ax.set_title("Monthly precipitation by Minnesota site (1927–1936)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Precipitation (inches)")

    # 5. Place the legend outside
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Site")

    # 6. Save the figure
    filepath = os.path.join(outdir, "minnesota_precip_line.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    return filepath
    # raise NotImplementedError("plot_minnesota_precip_line is not implemented yet.")


def main() -> List[str]:
    """Run all visualizations and return a list of generated file paths."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "output")
    ensure_output_dir(out_dir)
    figures: List[str] = []

    # Load and plot weather data
    weather_path = os.path.join(data_dir, "weather_data.csv")
    weather_df = pd.read_csv(weather_path)
    # Plot heatmap and scatter
    figures.append(plot_weather_heatmap(weather_df, out_dir))
    figures.append(plot_weather_scatter(weather_df, out_dir))

    # Load and plot global temperature anomalies
    global_path = os.path.join(data_dir, "global_temp.csv")
    global_df = pd.read_csv(global_path, skiprows=1)
    # Replace *** with NA and convert to numeric
    global_df = global_df.replace("***", pd.NA)
    for col in global_df.columns[1:]:
        global_df[col] = pd.to_numeric(global_df[col], errors="coerce")
    figures.append(plot_global_temp_heatmap(global_df, out_dir))

    # Load and plot Minnesota weather data
    minn_path = os.path.join(data_dir, "minnesota_weather.csv")
    minn_df = pd.read_csv(minn_path)
    figures.append(plot_minnesota_precip_line(minn_df, out_dir))
    return figures


if __name__ == "__main__":
    generated = main()
    print("Generated figures:")
    for path in generated:
        print(path)