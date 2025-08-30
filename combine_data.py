import pandas as pd
import os
from datetime import datetime

def combine_weekly_data(overlap_only=False):
    """
    Combine ICSA, gold, and crypto weekly data into a single dataset.
    All dates are normalized to YYYY-MM-DD format.
    Merges on week ending dates (Saturday).
    
    Args:
        overlap_only (bool): If True, only include weeks where all three datasets have data
    """
    
    # Define file paths
    icsa_file = "1_data_source_pinned/ICSA.csv"
    gold_file = "2_processed_data/gold_weekly.csv"
    crypto_file = "2_processed_data/crypto_weekly.csv"
    output_dir = "3_combined_data"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data files...")
    
    # Load ICSA data (already in weekly format with YYYY-MM-DD dates)
    icsa_df = pd.read_csv(icsa_file)
    icsa_df['observation_date'] = pd.to_datetime(icsa_df['observation_date'])
    
    # ICSA data appears to be on Saturdays, so we'll use it as-is
    icsa_df.rename(columns={'observation_date': 'week_ending_date'}, inplace=True)
    
    # Convert back to string format for consistent merging
    icsa_df['week_ending_date'] = icsa_df['week_ending_date'].dt.strftime('%Y-%m-%d')
    
    print(f"ICSA data: {len(icsa_df)} weeks")
    print(f"ICSA date range: {icsa_df['week_ending_date'].min()} to {icsa_df['week_ending_date'].max()}")
    
    # Check if processed files exist, if not, process them first
    if not os.path.exists(gold_file):
        print("\nGold weekly data not found. Processing gold data first...")
        from process_gold import process_gold_to_weekly
        process_gold_to_weekly()
    
    if not os.path.exists(crypto_file):
        print("\nCrypto weekly data not found. Processing crypto data first...")
        from process_crypto import process_crypto_to_weekly
        process_crypto_to_weekly()
    
    # Load gold weekly data
    gold_df = pd.read_csv(gold_file)
    print(f"\nGold data: {len(gold_df)} weeks")
    print(f"Gold date range: {gold_df['week_ending_date'].min()} to {gold_df['week_ending_date'].max()}")
    
    # Load crypto weekly data
    crypto_df = pd.read_csv(crypto_file)
    # Rename crypto columns to avoid confusion
    crypto_df.rename(columns={
        'market_cap': 'crypto_market_cap',
        'total_volume': 'crypto_total_volume'
    }, inplace=True)
    print(f"\nCrypto data: {len(crypto_df)} weeks")
    print(f"Crypto date range: {crypto_df['week_ending_date'].min()} to {crypto_df['week_ending_date'].max()}")
    
    # Merge all datasets on week_ending_date
    print("\nMerging datasets...")
    
    # First merge ICSA with gold
    combined_df = pd.merge(
        icsa_df,
        gold_df,
        on='week_ending_date',
        how='outer',
        suffixes=('', '_gold')
    )
    
    # Then merge with crypto
    combined_df = pd.merge(
        combined_df,
        crypto_df,
        on='week_ending_date',
        how='outer',
        suffixes=('', '_crypto')
    )
    
    # Sort by date
    combined_df['week_ending_date'] = pd.to_datetime(combined_df['week_ending_date'])
    combined_df.sort_values('week_ending_date', inplace=True)
    
    # Convert date back to string format
    combined_df['week_ending_date'] = combined_df['week_ending_date'].dt.strftime('%Y-%m-%d')
    
    # Save to CSV - choose filename based on overlap_only setting
    if overlap_only:
        output_file = os.path.join(output_dir, "final_data_overlap_only.csv")
        # Filter to only include rows where all three datasets have data
        combined_df = combined_df[
            combined_df['ICSA'].notna() & 
            combined_df['gold_close_avg'].notna() & 
            combined_df['crypto_market_cap'].notna()
        ]
        print(f"\nFiltered to overlap-only data: {len(combined_df)} weeks")
    else:
        output_file = os.path.join(output_dir, "final_data.csv")
    
    combined_df.to_csv(output_file, index=False)
    print(f"\nCombined data saved to: {output_file}")
    
    # Display summary
    print(f"\nCombined dataset: {len(combined_df)} weeks")
    print(f"Date range: {combined_df['week_ending_date'].min()} to {combined_df['week_ending_date'].max()}")
    
    # Show data availability
    icsa_available = combined_df['ICSA'].notna().sum()
    gold_available = combined_df['gold_close_avg'].notna().sum()
    crypto_available = combined_df['crypto_market_cap'].notna().sum()
    
    print(f"\nData availability:")
    print(f"  ICSA data: {icsa_available} weeks ({icsa_available/len(combined_df)*100:.1f}%)")
    print(f"  Gold data: {gold_available} weeks ({gold_available/len(combined_df)*100:.1f}%)")
    print(f"  Crypto data: {crypto_available} weeks ({crypto_available/len(combined_df)*100:.1f}%)")
    
    # Find overlapping period
    overlap_df = combined_df[
        combined_df['ICSA'].notna() & 
        combined_df['gold_close_avg'].notna() & 
        combined_df['crypto_market_cap'].notna()
    ]
    
    if len(overlap_df) > 0:
        print(f"\nOverlapping period with all data available:")
        print(f"  {len(overlap_df)} weeks")
        print(f"  From {overlap_df['week_ending_date'].min()} to {overlap_df['week_ending_date'].max()}")
    else:
        print("\nNo overlapping period found with all three datasets.")
    
    print("\nFirst 5 rows of combined data:")
    print(combined_df.head())
    
    print("\nLast 5 rows of combined data:")
    print(combined_df.tail())
    
    print("\nColumn names in final dataset:")
    print(list(combined_df.columns))
    
    return combined_df

def create_both_datasets():
    """Create both full and overlap-only datasets"""
    print("=== Creating Full Dataset (with missing data) ===")
    full_df = combine_weekly_data(overlap_only=False)
    
    print("\n=== Creating Overlap-Only Dataset (complete data only) ===")
    overlap_df = combine_weekly_data(overlap_only=True)
    
    return full_df, overlap_df

if __name__ == "__main__":
    create_both_datasets()