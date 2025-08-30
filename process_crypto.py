import pandas as pd
from datetime import datetime
import os

def process_crypto_to_weekly():
    """
    Convert CoinGecko crypto market cap data from daily snapshots to weekly averages.
    Timestamps are in Unix milliseconds format.
    Aligns to week ending on Saturday to match ICSA data format.
    """
    
    # Define file paths
    input_file = "1_data_source_pinned/crypto_market_cap.csv"
    output_dir = "2_processed_data"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading crypto market cap data...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert Unix millisecond timestamps to datetime
    df['date'] = pd.to_datetime(df['snapped_at'], unit='ms')
    
    # Normalize date to YYYY-MM-DD format
    df['date_normalized'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Set date as index for resampling
    df.set_index('date', inplace=True)
    
    print(f"Original data: {len(df)} daily snapshots")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Resample to weekly frequency (W-SAT means week ending on Saturday)
    # Using mean for prices/market cap, sum for volumes
    weekly_df = df.resample('W-SAT').agg({
        'market_cap': 'mean',  # Average market cap for the week
        'total_volume': 'sum'   # Total volume for the week
    }).reset_index()
    
    # Rename date column to match expected format
    weekly_df.rename(columns={'date': 'week_ending_date'}, inplace=True)
    
    # Format date as YYYY-MM-DD string
    weekly_df['week_ending_date'] = weekly_df['week_ending_date'].dt.strftime('%Y-%m-%d')
    
    # Remove rows with NaN values (in case of incomplete weeks)
    weekly_df = weekly_df.dropna()
    
    # Round numerical values for cleaner output
    weekly_df['market_cap'] = weekly_df['market_cap'].round(2)
    weekly_df['total_volume'] = weekly_df['total_volume'].round(2)
    
    # Save to CSV
    output_file = os.path.join(output_dir, "crypto_weekly.csv")
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
    print(f"Average weekly market cap: ${weekly_df['market_cap'].mean():,.2f}")
    print(f"Average weekly volume: ${weekly_df['total_volume'].mean():,.2f}")
    
    return weekly_df

if __name__ == "__main__":
    process_crypto_to_weekly()
