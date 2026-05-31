import os
import pandas as pd
import numpy as np

def generate_regression_data():
    np.random.seed(42)
    n_samples = 250
    
    square_feet = np.random.normal(1800, 500, n_samples).round().astype(int)
    # Clip square feet to a minimum of 600
    square_feet = np.clip(square_feet, 600, 4500)
    
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.1, 0.25, 0.4, 0.2, 0.05])
    bathrooms = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        # Bathrooms usually correlates with bedrooms
        bathrooms[i] = np.random.choice([1, 2, 3], p=[0.4, 0.5, 0.1]) if bedrooms[i] <= 2 else np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
        
    age_of_house = np.random.randint(0, 80, n_samples)
    distance_to_center = np.random.uniform(0.5, 25.0, n_samples).round(1)
    has_garage = np.random.choice(["Yes", "No"], size=n_samples, p=[0.7, 0.3])
    
    # Target variable: Price
    # Base price: $80,000
    # + $165 per sqft
    # + $22,000 per bedroom
    # + $15,000 per bathroom
    # - $950 per year of age
    # - $3,500 per mile from downtown
    # + $25,000 if it has a garage
    # + random gaussian noise
    noise = np.random.normal(0, 15000, n_samples)
    garage_multiplier = np.where(has_garage == "Yes", 25000, 0)
    
    price = (80000 + 165 * square_feet + 22000 * bedrooms + 15000 * bathrooms - 
             950 * age_of_house - 3500 * distance_to_center + garage_multiplier + noise)
    price = np.clip(price, 50000, None).round().astype(int)
    
    # Introduce a few missing values in DistanceToCenter and HasGarage to demonstrate cleaning!
    df = pd.DataFrame({
        "SquareFeet": square_feet,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "AgeOfHouse": age_of_house,
        "DistanceToCenter": distance_to_center,
        "HasGarage": has_garage,
        "Price": price
    })
    
    # Inject 5% missing values
    df.loc[df.sample(frac=0.04, random_state=10).index, "DistanceToCenter"] = np.nan
    df.loc[df.sample(frac=0.03, random_state=20).index, "HasGarage"] = np.nan
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/sample_housing.csv", index=False)
    print("Generated data/sample_housing.csv (Regression)")

def generate_timeseries_data():
    np.random.seed(42)
    dates = pd.date_range(start="2021-01-01", end="2025-12-01", freq="MS")
    n_samples = len(dates)
    
    # Trend
    trend = 120 * np.arange(n_samples)
    
    # Seasonality (12-month period)
    # Peak sales in summer (June/July) and winter (December)
    month_indices = dates.month
    seasonality = 1500 * np.sin(2 * np.pi * month_indices / 12) + 800 * np.cos(4 * np.pi * month_indices / 12)
    
    # Exogenous feature: MarketingSpend
    marketing_spend = np.random.normal(3000, 500, n_samples)
    marketing_spend = np.clip(marketing_spend, 1000, 6000).round().astype(int)
    
    # Base sales + trend + seasonality + marketing effect + noise
    noise = np.random.normal(0, 300, n_samples)
    sales = 8000 + trend + seasonality + 0.8 * marketing_spend + noise
    sales = sales.round().astype(int)
    
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "MarketingSpend": marketing_spend,
        "Sales": sales
    })
    
    # Inject a couple of missing values in MarketingSpend to test cleaning
    df.loc[df.sample(frac=0.03, random_state=30).index, "MarketingSpend"] = np.nan
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/sample_sales.csv", index=False)
    print("Generated data/sample_sales.csv (Time-Series)")

if __name__ == "__main__":
    generate_regression_data()
    generate_timeseries_data()
