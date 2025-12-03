# scripts/weather_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_inspect(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData info:")
    print(df.info())

    print("\nBasic statistics:")
    print(df.describe())

    return df

def clean_data(df):
    print("\nCleaning data...")

    # Drop rows with missing essential data and create a copy to avoid SettingWithCopyWarning
    df = df.dropna(subset=['Date', 'Temperature', 'Rainfall', 'Humidity']).copy()

    # Convert Date column to datetime format safely
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Drop rows where date conversion failed
    df = df.dropna(subset=['Date']).copy()

    # Reset index after drops
    df = df.reset_index(drop=True)

    print("\nData after cleaning:")
    print(df.info())

    return df

def compute_statistics(df):
    print("\nComputing statistics...")

    # Add columns for year and month for grouping
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Daily stats - mean, min, max (example with Temperature)
    daily_mean = df['Temperature'].mean()
    daily_min = df['Temperature'].min()
    daily_max = df['Temperature'].max()

    print(f"Temperature - Mean: {daily_mean:.2f}, Min: {daily_min:.2f}, Max: {daily_max:.2f}")

    # Monthly rainfall total
    monthly_rainfall = df.groupby('Month')['Rainfall'].sum()
    print("\nMonthly Rainfall Totals:")
    print(monthly_rainfall)

    return monthly_rainfall

def create_plots(df, monthly_rainfall, output_dir):
    print("\nCreating plots...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Line chart for daily temperature trends
    plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df['Temperature'], label='Temperature')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.title('Daily Temperature Trend')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'daily_temperature.png'))
    plt.close()

    # 2. Bar chart for monthly rainfall totals
    plt.figure(figsize=(8,5))
    monthly_rainfall.plot(kind='bar', color='skyblue')
    plt.xlabel('Month')
    plt.ylabel('Total Rainfall')
    plt.title('Monthly Rainfall Totals')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_rainfall.png'))
    plt.close()

    # 3. Scatter plot for humidity vs temperature
    plt.figure(figsize=(8,5))
    plt.scatter(df['Humidity'], df['Temperature'], alpha=0.5)
    plt.xlabel('Humidity')
    plt.ylabel('Temperature')
    plt.title('Humidity vs Temperature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'humidity_vs_temperature.png'))
    plt.close()

    # 4. Combined plot example (Temperature and Rainfall)
    fig, ax1 = plt.subplots(figsize=(10,5))

    ax1.plot(df['Date'], df['Temperature'], color='red', label='Temperature')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature', color='red')

    ax2 = ax1.twinx()
    ax2.bar(df['Date'], df['Rainfall'], color='blue', alpha=0.3, label='Rainfall')
    ax2.set_ylabel('Rainfall', color='blue')

    plt.title('Temperature and Rainfall Over Time')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temp_rainfall_combined.png'))
    plt.close()

    print("Plots saved to", output_dir)

def export_cleaned_data(df, output_path):
    print("\nExporting cleaned data...")
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    data_path = "../data/weather_data.csv"  # Adjust if needed
    output_folder = "../outputs/plots"
    cleaned_data_path = "../outputs/cleaned_data.csv"

    # Step 1: Load and inspect
    data = load_and_inspect(data_path)

    # Step 2: Clean data
    clean_data = clean_data(data)

    # Step 3: Compute statistics
    monthly_rain = compute_statistics(clean_data)

    # Step 4: Create plots
    create_plots(clean_data, monthly_rain, output_folder)

    # Step 5: Export cleaned data
    export_cleaned_data(clean_data, cleaned_data_path)

