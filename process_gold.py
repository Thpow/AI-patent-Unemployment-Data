import pandas as pd
from datetime import datetime
import os

def process_gold_to_weekly():
    """
    Convert gold price data from daily to weekly averages.
    Input date format: mm/dd/yyyy
    Output date format: yyyy-mm-dd (aligned to week ending Saturday)
    """
    
    # Define file paths
    input_file = "1_data_source_pinned/gold_historic.csv"
    output_dir = "2_processed_data"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading gold price data...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert date from mm/dd/yyyy to datetime
    df['date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    # Sort by date (important for resampling)
    df.sort_values('date', inplace=True)
    
    # Set date as index for resampling
    df.set_index('date', inplace=True)
    
    print(f"Original data: {len(df)} daily records")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Convert price columns to numeric (remove any commas if present)
    price_columns = ['Close/Last', 'Open', 'High', 'Low']
    for col in price_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    # Convert volume to numeric
    if 'Volume' in df.columns:
        df['Volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(',', ''), errors='coerce')
    
    # Resample to weekly frequency (W-SAT means week ending on Saturday)
    weekly_df = df.resample('W-SAT').agg({
        'Close/Last': 'mean',  # Average closing price
        'Open': 'first',       # First opening price of the week
        'High': 'max',         # Highest price of the week
        'Low': 'min',          # Lowest price of the week
        'Volume': 'sum'        # Total volume for the week
    }).reset_index()
    
    # Rename columns for clarity
    weekly_df.rename(columns={
        'date': 'week_ending_date',
        'Close/Last': 'gold_close_avg',
        'Open': 'gold_open',
        'High': 'gold_high',
        'Low': 'gold_low',
        'Volume': 'gold_volume'
    }, inplace=True)
    
    # Format date as YYYY-MM-DD string
    weekly_df['week_ending_date'] = weekly_df['week_ending_date'].dt.strftime('%Y-%m-%d')
    
    # Remove rows with NaN values
    weekly_df = weekly_df.dropna()
    
    # Round numerical values for cleaner output
    for col in ['gold_close_avg', 'gold_open', 'gold_high', 'gold_low']:
        weekly_df[col] = weekly_df[col].round(2)
    weekly_df['gold_volume'] = weekly_df['gold_volume'].round(0)
    
    # Save to CSV
    output_file = os.path.join(output_dir, "gold_weekly.csv")
    weekly_df.to_csv(output_file, index=False)
    print(f"\nWeekly data saved to: {output_file}")
    
    # Display summary
    print(f"\nProcessed {len(weekly_df)} weeks of data")
    print("\nFirst 5 weeks:")
    print(weekly_df.head())
    print("\nLast 5 weeks:")
    print(weekly_df.tail())
    
    # Show weekly statistics
    print("\nWeekly Statistics:")
    print(f"Average weekly gold price: ${weekly_df['gold_close_avg'].mean():,.2f}")
    print(f"Average weekly volume: {weekly_df['gold_volume'].mean():,.0f}")
    
    return weekly_df

if __name__ == "__main__":
    process_gold_to_weekly()