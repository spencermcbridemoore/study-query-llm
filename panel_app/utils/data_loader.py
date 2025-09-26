"""Data loading utilities"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_sample_data():
    """Load sample data for the application"""
    
    # Check if sample data exists
    data_path = Path(__file__).parent.parent.parent / "data" / "sample_data.csv"
    
    if data_path.exists():
        try:
            return pd.read_csv(
                data_path,
                comment="#",  # skips instructional comment lines
            )
        except Exception as e:
            print(f"Error loading data from {data_path}: {e}")
    
    # Generate sample data if file doesn't exist
    return generate_sample_data()


def generate_sample_data(n_rows=100):
    """Generate sample data for demonstration"""
    
    np.random.seed(42)
    
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="D")
    
    data = pd.DataFrame({
        "date": dates,
        "sales": np.random.exponential(1000, n_rows) + np.random.normal(0, 100, n_rows),
        "customers": np.random.poisson(50, n_rows),
        "revenue": np.random.gamma(2, 500, n_rows),
        "costs": np.random.uniform(200, 800, n_rows),
        "satisfaction": np.random.beta(8, 2, n_rows) * 100,
        "region": np.random.choice(["North", "South", "East", "West"], n_rows),
        "product": np.random.choice(["Product A", "Product B", "Product C"], n_rows),
    })
    
    # Calculate profit
    data["profit"] = data["revenue"] - data["costs"]
    
    # Round numeric columns
    numeric_cols = ["sales", "revenue", "costs", "profit", "satisfaction"]
    data[numeric_cols] = data[numeric_cols].round(2)
    
    return data


def save_sample_data(df, path=None):
    """Save sample data to CSV"""
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "sample_data.csv"
    
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Sample data saved to {path}")


if __name__ == "__main__":
    # Generate and save sample data when run directly
    sample_data = generate_sample_data()
    save_sample_data(sample_data)
    print(sample_data.head())
