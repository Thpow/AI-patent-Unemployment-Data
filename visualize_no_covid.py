import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load the no-COVID dataset and prepare it for analysis"""
    
    # Load the overlap-only dataset without COVID
    data_file = "3_combined_data/final_data_overlap_only_no_covid.csv"
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run combine_data_no_covid.py first.")
        return None
    
    df = pd.read_csv(data_file)
    df['week_ending_date'] = pd.to_datetime(df['week_ending_date'])
    
    print(f"Loaded dataset: {len(df)} weeks from {df['week_ending_date'].min()} to {df['week_ending_date'].max()}")
    print(f"No COVID period data (2020-2021 excluded)")
    
    # Create log transformations for better visualization
    df['log_icsa'] = np.log(df['ICSA'])
    df['log_gold'] = np.log(df['gold_close_avg'])
    df['log_crypto'] = np.log(df['crypto_market_cap'])
    
    # Create normalized versions (0-1 scale) for comparison
    df['norm_icsa'] = (df['ICSA'] - df['ICSA'].min()) / (df['ICSA'].max() - df['ICSA'].min())
    df['norm_gold'] = (df['gold_close_avg'] - df['gold_close_avg'].min()) / (df['gold_close_avg'].max() - df['gold_close_avg'].min())
    df['norm_crypto'] = (df['crypto_market_cap'] - df['crypto_market_cap'].min()) / (df['crypto_market_cap'].max() - df['crypto_market_cap'].min())
    
    return df

def create_time_series_analysis(df):
    """Create comprehensive time series plots without COVID distortion"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Economic Indicators Time Series Analysis (No COVID Period)', fontsize=16, fontweight='bold')
    
    # 1. Raw time series
    axes[0,0].plot(df['week_ending_date'], df['ICSA'], label='ICSA', color='red', alpha=0.8)
    axes[0,0].set_title('Unemployment Claims (ICSA) - Raw Values', fontweight='bold')
    axes[0,0].set_ylabel('Weekly Claims')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Gold prices
    axes[0,1].plot(df['week_ending_date'], df['gold_close_avg'], label='Gold Price', color='gold', alpha=0.8)
    axes[0,1].set_title('Gold Prices - Raw Values', fontweight='bold')
    axes[0,1].set_ylabel('Price ($)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Crypto market cap
    axes[1,0].plot(df['week_ending_date'], df['crypto_market_cap']/1e9, label='Crypto Market Cap', color='purple', alpha=0.8)
    axes[1,0].set_title('Cryptocurrency Market Cap - Raw Values', fontweight='bold')
    axes[1,0].set_ylabel('Market Cap (Billions $)')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Log scale comparison
    axes[1,1].plot(df['week_ending_date'], df['log_icsa'], label='Log(ICSA)', color='red', alpha=0.8)
    axes[1,1].plot(df['week_ending_date'], df['log_gold'], label='Log(Gold)', color='gold', alpha=0.8)
    axes[1,1].plot(df['week_ending_date'], df['log_crypto'], label='Log(Crypto)', color='purple', alpha=0.8)
    axes[1,1].set_title('Log Scale Comparison', fontweight='bold')
    axes[1,1].set_ylabel('Log Values')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # 5. Normalized comparison (0-1 scale)
    axes[2,0].plot(df['week_ending_date'], df['norm_icsa'], label='ICSA (normalized)', color='red', alpha=0.8)
    axes[2,0].plot(df['week_ending_date'], df['norm_gold'], label='Gold (normalized)', color='gold', alpha=0.8)
    axes[2,0].plot(df['week_ending_date'], df['norm_crypto'], label='Crypto (normalized)', color='purple', alpha=0.8)
    axes[2,0].set_title('Normalized Comparison (0-1 Scale)', fontweight='bold')
    axes[2,0].set_ylabel('Normalized Values')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)
    axes[2,0].tick_params(axis='x', rotation=45)
    
    # 6. Rolling correlations
    window = 52  # 1 year rolling window
    rolling_corr_gold = df['log_icsa'].rolling(window).corr(df['log_gold'])
    rolling_corr_crypto = df['log_icsa'].rolling(window).corr(df['log_crypto'])
    
    axes[2,1].plot(df['week_ending_date'], rolling_corr_gold, label='ICSA vs Gold', color='orange', alpha=0.8)
    axes[2,1].plot(df['week_ending_date'], rolling_corr_crypto, label='ICSA vs Crypto', color='purple', alpha=0.8)
    axes[2,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2,1].set_title('Rolling Correlations (52-week window)', fontweight='bold')
    axes[2,1].set_ylabel('Correlation')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)
    axes[2,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/time_series_no_covid.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_correlation_analysis(df):
    """Create correlation analysis without COVID distortion"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Correlation Analysis (No COVID Period)', fontsize=16, fontweight='bold')
    
    # 1. Correlation scatter: ICSA vs Gold
    axes[0,0].scatter(df['ICSA'], df['gold_close_avg'], alpha=0.6, color='orange', s=20)
    z = np.polyfit(df['ICSA'], df['gold_close_avg'], 1)
    p = np.poly1d(z)
    axes[0,0].plot(df['ICSA'], p(df['ICSA']), "r--", alpha=0.8, linewidth=2)
    axes[0,0].set_xlabel('ICSA (Weekly Claims)')
    axes[0,0].set_ylabel('Gold Price ($)')
    axes[0,0].set_title('Unemployment vs Gold Price', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    corr_icsa_gold = df['ICSA'].corr(df['gold_close_avg'])
    axes[0,0].text(0.05, 0.95, f'Correlation: {corr_icsa_gold:.3f}', 
                   transform=axes[0,0].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 2. Correlation scatter: ICSA vs Crypto
    axes[0,1].scatter(df['ICSA'], df['crypto_market_cap']/1e9, alpha=0.6, color='purple', s=20)
    z = np.polyfit(df['ICSA'], df['crypto_market_cap']/1e9, 1)
    p = np.poly1d(z)
    axes[0,1].plot(df['ICSA'], p(df['ICSA']), "r--", alpha=0.8, linewidth=2)
    axes[0,1].set_xlabel('ICSA (Weekly Claims)')
    axes[0,1].set_ylabel('Crypto Market Cap (Billions $)')
    axes[0,1].set_title('Unemployment vs Crypto Market Cap', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    corr_icsa_crypto = df['ICSA'].corr(df['crypto_market_cap'])
    axes[0,1].text(0.05, 0.95, f'Correlation: {corr_icsa_crypto:.3f}', 
                   transform=axes[0,1].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 3. Log correlation scatter: ICSA vs Gold
    axes[1,0].scatter(df['log_icsa'], df['log_gold'], alpha=0.6, color='orange', s=20)
    z = np.polyfit(df['log_icsa'], df['log_gold'], 1)
    p = np.poly1d(z)
    axes[1,0].plot(df['log_icsa'], p(df['log_icsa']), "r--", alpha=0.8, linewidth=2)
    axes[1,0].set_xlabel('Log(ICSA)')
    axes[1,0].set_ylabel('Log(Gold Price)')
    axes[1,0].set_title('Log Scale: Unemployment vs Gold', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    log_corr_icsa_gold = df['log_icsa'].corr(df['log_gold'])
    axes[1,0].text(0.05, 0.95, f'Log Correlation: {log_corr_icsa_gold:.3f}', 
                   transform=axes[1,0].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 4. Correlation matrix heatmap
    corr_vars = ['ICSA', 'gold_close_avg', 'crypto_market_cap']
    corr_matrix = df[corr_vars].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=axes[1,1], cbar_kws={'label': 'Correlation'})
    axes[1,1].set_title('Correlation Matrix', fontweight='bold')
    
    # Rename labels for better readability
    axes[1,1].set_xticklabels(['ICSA', 'Gold Price', 'Crypto Market Cap'])
    axes[1,1].set_yticklabels(['ICSA', 'Gold Price', 'Crypto Market Cap'])
    
    plt.tight_layout()
    plt.savefig('visualizations/correlation_analysis_no_covid.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_matrix

def create_volatility_analysis(df):
    """Analyze volatility patterns without COVID distortion"""
    
    # Calculate rolling volatility (standard deviation)
    window = 26  # 6-month rolling window
    
    df['icsa_volatility'] = df['ICSA'].rolling(window).std()
    df['gold_volatility'] = df['gold_close_avg'].rolling(window).std()
    df['crypto_volatility'] = df['crypto_market_cap'].rolling(window).std()
    
    # Calculate percentage changes
    df['icsa_pct_change'] = df['ICSA'].pct_change()
    df['gold_pct_change'] = df['gold_close_avg'].pct_change()
    df['crypto_pct_change'] = df['crypto_market_cap'].pct_change()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Volatility Analysis (No COVID Period)', fontsize=16, fontweight='bold')
    
    # 1. Rolling volatility
    axes[0,0].plot(df['week_ending_date'], df['icsa_volatility'], label='ICSA Volatility', color='red', alpha=0.8)
    axes[0,0].set_title('ICSA Rolling Volatility (26-week window)', fontweight='bold')
    axes[0,0].set_ylabel('Standard Deviation')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Gold volatility
    axes[0,1].plot(df['week_ending_date'], df['gold_volatility'], label='Gold Volatility', color='gold', alpha=0.8)
    axes[0,1].set_title('Gold Price Rolling Volatility (26-week window)', fontweight='bold')
    axes[0,1].set_ylabel('Standard Deviation')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Crypto volatility
    axes[1,0].plot(df['week_ending_date'], df['crypto_volatility']/1e9, label='Crypto Volatility', color='purple', alpha=0.8)
    axes[1,0].set_title('Crypto Market Cap Rolling Volatility (26-week window)', fontweight='bold')
    axes[1,0].set_ylabel('Standard Deviation (Billions $)')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Percentage change distributions
    axes[1,1].hist(df['icsa_pct_change'].dropna(), bins=50, alpha=0.7, label='ICSA', color='red', density=True)
    axes[1,1].hist(df['gold_pct_change'].dropna(), bins=50, alpha=0.7, label='Gold', color='gold', density=True)
    axes[1,1].hist(df['crypto_pct_change'].dropna(), bins=50, alpha=0.7, label='Crypto', color='purple', density=True)
    axes[1,1].set_title('Weekly Percentage Change Distributions', fontweight='bold')
    axes[1,1].set_xlabel('Percentage Change')
    axes[1,1].set_ylabel('Density')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/volatility_analysis_no_covid.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print volatility statistics
    print("\n=== Volatility Statistics (No COVID Period) ===")
    print(f"ICSA volatility - Mean: {df['icsa_volatility'].mean():.0f}, Std: {df['icsa_volatility'].std():.0f}")
    print(f"Gold volatility - Mean: {df['gold_volatility'].mean():.2f}, Std: {df['gold_volatility'].std():.2f}")
    print(f"Crypto volatility - Mean: {df['crypto_volatility'].mean()/1e9:.2f}B, Std: {df['crypto_volatility'].std()/1e9:.2f}B")
    
    print(f"\nPercentage Change Statistics:")
    print(f"ICSA - Mean: {df['icsa_pct_change'].mean()*100:.2f}%, Std: {df['icsa_pct_change'].std()*100:.2f}%")
    print(f"Gold - Mean: {df['gold_pct_change'].mean()*100:.2f}%, Std: {df['gold_pct_change'].std()*100:.2f}%")
    print(f"Crypto - Mean: {df['crypto_pct_change'].mean()*100:.2f}%, Std: {df['crypto_pct_change'].std()*100:.2f}%")

def create_recent_period_analysis(df):
    """Create focused analysis for recent period (2022+) without COVID distortion"""
    
    # Filter data for 2022 onwards (post-COVID recovery)
    recent_data = df[df['week_ending_date'] >= '2022-01-01'].copy()
    
    if len(recent_data) == 0:
        print("No data available for 2022+")
        return
    
    print(f"\n=== Recent Period Analysis (2022+): {len(recent_data)} weeks ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Recent Period Analysis (2022+ Post-COVID Recovery)', fontsize=16, fontweight='bold')
    
    # 1. Recent time series with emphasized unemployment
    axes[0,0].plot(recent_data['week_ending_date'], recent_data['norm_icsa'] * 3, 
                   label='ICSA (3x amplified)', color='red', linewidth=2, alpha=0.9)
    axes[0,0].plot(recent_data['week_ending_date'], recent_data['norm_gold'], 
                   label='Gold (normalized)', color='gold', alpha=0.8)
    axes[0,0].plot(recent_data['week_ending_date'], recent_data['norm_crypto'], 
                   label='Crypto (normalized)', color='purple', alpha=0.8)
    axes[0,0].set_title('Recent Trends - Unemployment Movements Emphasized', fontweight='bold')
    axes[0,0].set_ylabel('Normalized Values')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Recent correlations
    recent_corr_gold = recent_data['log_icsa'].corr(recent_data['log_gold'])
    recent_corr_crypto = recent_data['log_icsa'].corr(recent_data['log_crypto'])
    
    correlations = ['ICSA vs Gold', 'ICSA vs Crypto']
    values = [recent_corr_gold, recent_corr_crypto]
    colors = ['orange', 'purple']
    
    bars = axes[0,1].bar(correlations, values, color=colors, alpha=0.7)
    axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0,1].set_title('Recent Period Correlations (2022+)', fontweight='bold')
    axes[0,1].set_ylabel('Correlation Coefficient')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.03,
                       f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # 3. Recent volatility comparison
    recent_data['icsa_vol'] = recent_data['ICSA'].rolling(13).std()  # 3-month window
    recent_data['gold_vol'] = recent_data['gold_close_avg'].rolling(13).std()
    recent_data['crypto_vol'] = recent_data['crypto_market_cap'].rolling(13).std()
    
    # Normalize volatilities for comparison
    icsa_vol_norm = recent_data['icsa_vol'] / recent_data['icsa_vol'].max()
    gold_vol_norm = recent_data['gold_vol'] / recent_data['gold_vol'].max()
    crypto_vol_norm = recent_data['crypto_vol'] / recent_data['crypto_vol'].max()
    
    axes[1,0].plot(recent_data['week_ending_date'], icsa_vol_norm, 
                   label='ICSA Volatility', color='red', alpha=0.8)
    axes[1,0].plot(recent_data['week_ending_date'], gold_vol_norm, 
                   label='Gold Volatility', color='gold', alpha=0.8)
    axes[1,0].plot(recent_data['week_ending_date'], crypto_vol_norm, 
                   label='Crypto Volatility', color='purple', alpha=0.8)
    axes[1,0].set_title('Recent Volatility Comparison (Normalized)', fontweight='bold')
    axes[1,0].set_ylabel('Normalized Volatility')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Reactivity analysis - how gold/crypto react to unemployment changes
    recent_data['icsa_change'] = recent_data['ICSA'].pct_change()
    recent_data['gold_change'] = recent_data['gold_close_avg'].pct_change()
    recent_data['crypto_change'] = recent_data['crypto_market_cap'].pct_change()
    
    # Calculate reactivity (correlation of changes)
    gold_reactivity = recent_data['icsa_change'].corr(recent_data['gold_change'])
    crypto_reactivity = recent_data['icsa_change'].corr(recent_data['crypto_change'])
    
    reactivity_labels = ['Gold Reactivity', 'Crypto Reactivity']
    reactivity_values = [gold_reactivity, crypto_reactivity]
    reactivity_colors = ['gold', 'purple']
    
    bars = axes[1,1].bar(reactivity_labels, reactivity_values, color=reactivity_colors, alpha=0.7)
    axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,1].set_title('Asset Reactivity to Unemployment Changes', fontweight='bold')
    axes[1,1].set_ylabel('Change Correlation')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, reactivity_values):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.03,
                       f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/recent_period_analysis_no_covid.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print recent period statistics
    print(f"Recent period correlations:")
    print(f"  ICSA vs Gold: {recent_corr_gold:.3f}")
    print(f"  ICSA vs Crypto: {recent_corr_crypto:.3f}")
    print(f"Asset reactivity to unemployment changes:")
    print(f"  Gold reactivity: {gold_reactivity:.3f}")
    print(f"  Crypto reactivity: {crypto_reactivity:.3f}")

def main():
    """Main analysis function"""
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    print("=== Economic Data Analysis (No COVID Period) ===")
    
    # Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Create all visualizations
    print("\n1. Creating time series analysis...")
    create_time_series_analysis(df)
    
    print("\n2. Creating correlation analysis...")
    corr_matrix = create_correlation_analysis(df)
    
    print("\n3. Creating volatility analysis...")
    create_volatility_analysis(df)
    
    print("\n4. Creating recent period analysis...")
    create_recent_period_analysis(df)
    
    print("\n=== Analysis Complete ===")
    print("All visualizations saved to visualizations/ directory:")
    print("- time_series_no_covid.png")
    print("- correlation_analysis_no_covid.png") 
    print("- volatility_analysis_no_covid.png")
    print("- recent_period_analysis_no_covid.png")
    
    return df, corr_matrix

if __name__ == "__main__":
    df, corr_matrix = main()
