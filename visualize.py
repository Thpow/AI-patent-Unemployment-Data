import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load the overlap-only dataset and prepare it for analysis"""
    
    # Load the clean dataset
    df = pd.read_csv('3_combined_data/final_data_overlap_only.csv')
    
    # Convert date to datetime
    df['week_ending_date'] = pd.to_datetime(df['week_ending_date'])
    
    # Create additional derived features
    df['year'] = df['week_ending_date'].dt.year
    df['month'] = df['week_ending_date'].dt.month
    df['quarter'] = df['week_ending_date'].dt.quarter
    
    # Calculate percentage changes for trend analysis
    df['icsa_pct_change'] = df['ICSA'].pct_change() * 100
    df['gold_pct_change'] = df['gold_close_avg'].pct_change() * 100
    df['crypto_pct_change'] = df['crypto_market_cap'].pct_change() * 100
    
    # Calculate rolling averages to smooth volatility
    df['icsa_ma_4w'] = df['ICSA'].rolling(window=4).mean()
    df['gold_ma_4w'] = df['gold_close_avg'].rolling(window=4).mean()
    df['crypto_ma_4w'] = df['crypto_market_cap'].rolling(window=4).mean()
    
    # Create log-transformed variables for better visualization
    df['log_icsa'] = np.log(df['ICSA'])
    df['log_gold'] = np.log(df['gold_close_avg'])
    df['log_crypto'] = np.log(df['crypto_market_cap'])
    
    # Enhanced log transformations for extreme outliers (COVID spike)
    # Double log for ICSA to handle extreme COVID values
    df['log2_icsa'] = np.log(np.log(df['ICSA'] + 1) + 1)  # Double log with safety offset
    
    # Inverse hyperbolic sine (asinh) - better for data with zeros and extreme values
    df['asinh_icsa'] = np.arcsinh(df['ICSA'] / 1000)  # Scale down first
    df['asinh_gold'] = np.arcsinh(df['gold_close_avg'])
    df['asinh_crypto'] = np.arcsinh(df['crypto_market_cap'] / 1e12)  # Scale to trillions
    
    # Box-Cox style transformation (manual lambda tuning for ICSA)
    lambda_icsa = 0.1  # Very small lambda for strong transformation
    df['boxcox_icsa'] = (np.power(df['ICSA'], lambda_icsa) - 1) / lambda_icsa
    
    # Log-transformed moving averages
    df['log_icsa_ma_4w'] = np.log(df['icsa_ma_4w'])
    df['log_gold_ma_4w'] = np.log(df['gold_ma_4w'])
    df['log_crypto_ma_4w'] = np.log(df['crypto_ma_4w'])
    
    # Enhanced log moving averages
    df['log2_icsa_ma_4w'] = np.log(np.log(df['icsa_ma_4w'] + 1) + 1)
    df['asinh_icsa_ma_4w'] = np.arcsinh(df['icsa_ma_4w'] / 1000)
    
    # Create unemployment stress indicator (higher ICSA = more stress)
    df['unemployment_stress'] = (df['ICSA'] - df['ICSA'].mean()) / df['ICSA'].std()
    
    print(f"Dataset loaded: {len(df)} weeks from {df['week_ending_date'].min().date()} to {df['week_ending_date'].max().date()}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def create_time_series_analysis(df):
    """Create comprehensive time series visualizations with enhanced log scales"""
    
    fig, axes = plt.subplots(4, 2, figsize=(20, 18))
    fig.suptitle('Economic Indicators: Enhanced Log Scale Analysis for COVID Outliers', fontsize=16, fontweight='bold')
    
    # Highlight COVID period
    covid_start = pd.to_datetime('2020-03-01')
    covid_end = pd.to_datetime('2021-06-01')
    
    # 1. ICSA - Linear Scale (for reference)
    axes[0,0].plot(df['week_ending_date'], df['ICSA'], color='red', alpha=0.7, linewidth=1)
    axes[0,0].plot(df['week_ending_date'], df['icsa_ma_4w'], color='darkred', linewidth=2, label='4-week MA')
    axes[0,0].set_title('ICSA - Linear Scale (COVID dominates)', fontweight='bold')
    axes[0,0].set_ylabel('Claims')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axvspan(covid_start, covid_end, alpha=0.2, color='red')
    
    # 2. ICSA - Double Log Scale (strongest transformation)
    axes[0,1].plot(df['week_ending_date'], df['log2_icsa'], color='red', alpha=0.7, linewidth=1)
    axes[0,1].plot(df['week_ending_date'], df['log2_icsa_ma_4w'], color='darkred', linewidth=2, label='4-week MA')
    axes[0,1].set_title('ICSA - Double Log Scale (log(log(x)))', fontweight='bold')
    axes[0,1].set_ylabel('Log(Log(Claims))')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axvspan(covid_start, covid_end, alpha=0.2, color='red')
    
    # 3. ICSA - Inverse Hyperbolic Sine (asinh)
    axes[1,0].plot(df['week_ending_date'], df['asinh_icsa'], color='red', alpha=0.7, linewidth=1)
    axes[1,0].plot(df['week_ending_date'], df['asinh_icsa_ma_4w'], color='darkred', linewidth=2, label='4-week MA')
    axes[1,0].set_title('ICSA - Inverse Hyperbolic Sine (asinh)', fontweight='bold')
    axes[1,0].set_ylabel('asinh(Claims/1000)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axvspan(covid_start, covid_end, alpha=0.2, color='red')
    
    # 4. ICSA - Box-Cox Style (power = 0.1)
    axes[1,1].plot(df['week_ending_date'], df['boxcox_icsa'], color='red', alpha=0.7, linewidth=1)
    axes[1,1].set_title('ICSA - Box-Cox Style (λ=0.1)', fontweight='bold')
    axes[1,1].set_ylabel('(x^0.1 - 1) / 0.1')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axvspan(covid_start, covid_end, alpha=0.2, color='red')
    
    # 5. Gold with enhanced scaling
    axes[2,0].plot(df['week_ending_date'], df['gold_close_avg'], color='gold', alpha=0.7, linewidth=1)
    axes[2,0].plot(df['week_ending_date'], df['gold_ma_4w'], color='orange', linewidth=2, label='4-week MA')
    axes[2,0].set_title('Gold Prices - Linear Scale', fontweight='bold')
    axes[2,0].set_ylabel('Price ($)')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)
    axes[2,0].axvspan(covid_start, covid_end, alpha=0.2, color='red')
    
    # 6. Gold - asinh Scale
    axes[2,1].plot(df['week_ending_date'], df['asinh_gold'], color='gold', alpha=0.7, linewidth=1)
    axes[2,1].set_title('Gold Prices - asinh Scale', fontweight='bold')
    axes[2,1].set_ylabel('asinh(Price)')
    axes[2,1].grid(True, alpha=0.3)
    axes[2,1].axvspan(covid_start, covid_end, alpha=0.2, color='red')
    
    # 7. Crypto - Linear Scale
    axes[3,0].plot(df['week_ending_date'], df['crypto_market_cap']/1e12, color='blue', alpha=0.7, linewidth=1)
    axes[3,0].plot(df['week_ending_date'], df['crypto_ma_4w']/1e12, color='darkblue', linewidth=2, label='4-week MA')
    axes[3,0].set_title('Crypto Market Cap - Linear Scale', fontweight='bold')
    axes[3,0].set_ylabel('Market Cap (Trillions $)')
    axes[3,0].legend()
    axes[3,0].grid(True, alpha=0.3)
    axes[3,0].axvspan(covid_start, covid_end, alpha=0.2, color='red')
    
    # 8. Crypto - asinh Scale
    axes[3,1].plot(df['week_ending_date'], df['asinh_crypto'], color='blue', alpha=0.7, linewidth=1)
    axes[3,1].set_title('Crypto Market Cap - asinh Scale', fontweight='bold')
    axes[3,1].set_ylabel('asinh(Market Cap/1T)')
    axes[3,1].grid(True, alpha=0.3)
    axes[3,1].axvspan(covid_start, covid_end, alpha=0.2, color='red')
    
    plt.tight_layout()
    plt.savefig('visualizations/time_series_linear_vs_log.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_log_comparison_analysis(df):
    """Create log-scale comparison and correlation analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Log-Scale Analysis: Better Visualization of Relationships', fontsize=16, fontweight='bold')
    
    # 1. All three series with enhanced log scaling for better comparison
    axes[0,0].plot(df['week_ending_date'], df['log2_icsa'], color='red', label='Double Log(ICSA)', linewidth=2, alpha=0.8)
    axes[0,0].plot(df['week_ending_date'], df['asinh_gold'], color='gold', label='asinh(Gold)', linewidth=2, alpha=0.8)
    axes[0,0].plot(df['week_ending_date'], df['asinh_crypto'], color='blue', label='asinh(Crypto)', linewidth=2, alpha=0.8)
    axes[0,0].set_title('Enhanced Log Scale Comparison - COVID Spike Normalized', fontweight='bold')
    axes[0,0].set_ylabel('Transformed Values')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Highlight COVID period
    covid_start = pd.to_datetime('2020-03-01')
    covid_end = pd.to_datetime('2021-06-01')
    axes[0,0].axvspan(covid_start, covid_end, alpha=0.2, color='red')
    
    # 2. Enhanced correlation scatter: Double Log ICSA vs asinh Gold
    axes[0,1].scatter(df['log2_icsa'], df['asinh_gold'], alpha=0.6, color='orange', s=20)
    z = np.polyfit(df['log2_icsa'].dropna(), df['asinh_gold'].dropna(), 1)
    p = np.poly1d(z)
    axes[0,1].plot(df['log2_icsa'], p(df['log2_icsa']), "r--", alpha=0.8, linewidth=2)
    axes[0,1].set_xlabel('Double Log(ICSA)')
    axes[0,1].set_ylabel('asinh(Gold Price)')
    axes[0,1].set_title('Enhanced Scale: Unemployment vs Gold', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    enhanced_corr_icsa_gold = df['log2_icsa'].corr(df['asinh_gold'])
    axes[0,1].text(0.05, 0.95, f'Enhanced Correlation: {enhanced_corr_icsa_gold:.3f}', 
                   transform=axes[0,1].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 3. Enhanced correlation scatter: Double Log ICSA vs asinh Crypto
    axes[1,0].scatter(df['log2_icsa'], df['asinh_crypto'], alpha=0.6, color='purple', s=20)
    z = np.polyfit(df['log2_icsa'].dropna(), df['asinh_crypto'].dropna(), 1)
    p = np.poly1d(z)
    axes[1,0].plot(df['log2_icsa'], p(df['log2_icsa']), "r--", alpha=0.8, linewidth=2)
    axes[1,0].set_xlabel('Double Log(ICSA)')
    axes[1,0].set_ylabel('asinh(Crypto Market Cap)')
    axes[1,0].set_title('Enhanced Scale: Unemployment vs Crypto', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    enhanced_corr_icsa_crypto = df['log2_icsa'].corr(df['asinh_crypto'])
    axes[1,0].text(0.05, 0.95, f'Enhanced Correlation: {enhanced_corr_icsa_crypto:.3f}', 
                   transform=axes[1,0].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 4. Enhanced correlation matrix heatmap
    enhanced_vars = ['log2_icsa', 'asinh_gold', 'asinh_crypto', 'boxcox_icsa']
    enhanced_corr_matrix = df[enhanced_vars].corr()
    
    sns.heatmap(enhanced_corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=axes[1,1], cbar_kws={'label': 'Enhanced Correlation'})
    axes[1,1].set_title('Enhanced Transform Correlation Matrix', fontweight='bold')
    
    # Rename labels for better readability
    axes[1,1].set_xticklabels(['Double Log(ICSA)', 'asinh(Gold)', 'asinh(Crypto)', 'BoxCox(ICSA)'])
    axes[1,1].set_yticklabels(['Double Log(ICSA)', 'asinh(Gold)', 'asinh(Crypto)', 'BoxCox(ICSA)'])
    
    plt.tight_layout()
    plt.savefig('visualizations/log_scale_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return enhanced_corr_matrix

def create_recent_period_analysis(df):
    """Create focused analysis for 2023+ with emphasized unemployment movements"""
    
    # Filter data for 2023 onwards
    recent_data = df[df['week_ending_date'] >= '2023-01-01'].copy()
    
    if len(recent_data) == 0:
        print("No data available for 2023+")
        return
    
    # Create emphasized unemployment movements by scaling
    # Method 1: Z-score normalization to emphasize relative movements
    recent_data['icsa_zscore'] = (recent_data['ICSA'] - recent_data['ICSA'].mean()) / recent_data['ICSA'].std()
    recent_data['gold_zscore'] = (recent_data['gold_close_avg'] - recent_data['gold_close_avg'].mean()) / recent_data['gold_close_avg'].std()
    recent_data['crypto_zscore'] = (recent_data['crypto_market_cap'] - recent_data['crypto_market_cap'].mean()) / recent_data['crypto_market_cap'].std()
    
    # Method 2: Amplify unemployment movements by scaling factor
    icsa_amplified = recent_data['icsa_zscore'] * 3  # Amplify by 3x for visibility
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle('Recent Period Analysis (2023+): Emphasized Unemployment Movements', fontsize=16, fontweight='bold')
    
    # 1. Raw values - recent period
    axes[0,0].plot(recent_data['week_ending_date'], recent_data['ICSA'], color='red', linewidth=2, label='ICSA')
    axes[0,0].set_title('Unemployment Claims (2023+) - Raw Values', fontweight='bold')
    axes[0,0].set_ylabel('Claims')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Emphasized unemployment with dual axis
    ax1 = axes[0,1]
    ax1.plot(recent_data['week_ending_date'], icsa_amplified, color='red', linewidth=3, label='ICSA (Amplified 3x)')
    ax1.set_ylabel('Amplified ICSA Z-Score', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_title('Amplified Unemployment Movements vs Assets', fontweight='bold')
    
    ax2 = ax1.twinx()
    ax2.plot(recent_data['week_ending_date'], recent_data['gold_zscore'], color='gold', linewidth=2, label='Gold Z-Score')
    ax2.plot(recent_data['week_ending_date'], recent_data['crypto_zscore'], color='blue', linewidth=2, label='Crypto Z-Score')
    ax2.set_ylabel('Asset Z-Scores', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # 3. All normalized on same scale for shape comparison
    axes[1,0].plot(recent_data['week_ending_date'], recent_data['icsa_zscore'], color='red', linewidth=3, label='ICSA (Normalized)', alpha=0.8)
    axes[1,0].plot(recent_data['week_ending_date'], recent_data['gold_zscore'], color='gold', linewidth=2, label='Gold (Normalized)', alpha=0.8)
    axes[1,0].plot(recent_data['week_ending_date'], recent_data['crypto_zscore'], color='blue', linewidth=2, label='Crypto (Normalized)', alpha=0.8)
    axes[1,0].set_title('Normalized Comparison - Shape Analysis', fontweight='bold')
    axes[1,0].set_ylabel('Z-Score (Standard Deviations)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 4. Rolling correlation over recent period
    window = 8  # 8-week rolling window
    recent_data['rolling_corr_icsa_gold'] = recent_data['ICSA'].rolling(window=window).corr(recent_data['gold_close_avg'])
    recent_data['rolling_corr_icsa_crypto'] = recent_data['ICSA'].rolling(window=window).corr(recent_data['crypto_market_cap'])
    
    axes[1,1].plot(recent_data['week_ending_date'], recent_data['rolling_corr_icsa_gold'], color='orange', linewidth=2, label='ICSA-Gold Correlation')
    axes[1,1].plot(recent_data['week_ending_date'], recent_data['rolling_corr_icsa_crypto'], color='purple', linewidth=2, label='ICSA-Crypto Correlation')
    axes[1,1].set_title(f'{window}-Week Rolling Correlations (2023+)', fontweight='bold')
    axes[1,1].set_ylabel('Correlation Coefficient')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 5. Volatility comparison - focus on crypto vs gold reactivity
    # Calculate rolling volatility (standard deviation of percentage changes)
    vol_window = 4  # 4-week rolling volatility
    recent_data['icsa_volatility'] = recent_data['icsa_pct_change'].rolling(window=vol_window).std()
    recent_data['gold_volatility'] = recent_data['gold_pct_change'].rolling(window=vol_window).std()
    recent_data['crypto_volatility'] = recent_data['crypto_pct_change'].rolling(window=vol_window).std()
    
    # Plot percentage changes with volatility bands
    axes[2,0].plot(recent_data['week_ending_date'], recent_data['gold_pct_change'], color='gold', linewidth=2, alpha=0.8, label='Gold % Change')
    axes[2,0].plot(recent_data['week_ending_date'], recent_data['crypto_pct_change'], color='blue', linewidth=2, alpha=0.8, label='Crypto % Change')
    
    # Add volatility bands
    axes[2,0].fill_between(recent_data['week_ending_date'], 
                          -recent_data['gold_volatility'], recent_data['gold_volatility'], 
                          color='gold', alpha=0.2, label='Gold Volatility Band')
    axes[2,0].fill_between(recent_data['week_ending_date'], 
                          -recent_data['crypto_volatility'], recent_data['crypto_volatility'], 
                          color='blue', alpha=0.2, label='Crypto Volatility Band')
    
    axes[2,0].set_title('Price Change Volatility: Crypto vs Gold (2023+)', fontweight='bold')
    axes[2,0].set_ylabel('% Change')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)
    axes[2,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add volatility statistics as text
    gold_vol_avg = recent_data['gold_volatility'].mean()
    crypto_vol_avg = recent_data['crypto_volatility'].mean()
    vol_ratio = crypto_vol_avg / gold_vol_avg if gold_vol_avg > 0 else 0
    
    axes[2,0].text(0.02, 0.98, f'Avg Volatility:\nGold: {gold_vol_avg:.2f}%\nCrypto: {crypto_vol_avg:.2f}%\nRatio: {vol_ratio:.1f}x', 
                   transform=axes[2,0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    # 6. Price change correlation with unemployment - which asset reacts more?
    # Calculate correlation of price changes with unemployment changes
    recent_data['icsa_pct_change_lag1'] = recent_data['icsa_pct_change'].shift(1)  # Lagged unemployment change
    
    # Rolling correlations between unemployment changes and asset price changes
    corr_window = 8
    recent_data['corr_icsa_gold_pct'] = recent_data['icsa_pct_change'].rolling(window=corr_window).corr(recent_data['gold_pct_change'])
    recent_data['corr_icsa_crypto_pct'] = recent_data['icsa_pct_change'].rolling(window=corr_window).corr(recent_data['crypto_pct_change'])
    
    # Plot correlations and reactivity analysis
    axes[2,1].plot(recent_data['week_ending_date'], recent_data['corr_icsa_gold_pct'], 
                   color='orange', linewidth=3, label='ICSA-Gold % Change Correlation', alpha=0.8)
    axes[2,1].plot(recent_data['week_ending_date'], recent_data['corr_icsa_crypto_pct'], 
                   color='purple', linewidth=3, label='ICSA-Crypto % Change Correlation', alpha=0.8)
    
    axes[2,1].set_title('Price Change Reactivity to Unemployment (2023+)', fontweight='bold')
    axes[2,1].set_ylabel('Rolling Correlation (% Changes)')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)
    axes[2,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[2,1].axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Strong Positive')
    axes[2,1].axhline(y=-0.3, color='red', linestyle='--', alpha=0.5, label='Strong Negative')
    
    # Calculate which asset is more reactive
    gold_reactivity = abs(recent_data['corr_icsa_gold_pct']).mean()
    crypto_reactivity = abs(recent_data['corr_icsa_crypto_pct']).mean()
    more_reactive = "Crypto" if crypto_reactivity > gold_reactivity else "Gold"
    reactivity_ratio = max(crypto_reactivity, gold_reactivity) / min(crypto_reactivity, gold_reactivity) if min(crypto_reactivity, gold_reactivity) > 0 else 0
    
    axes[2,1].text(0.02, 0.98, f'Reactivity to Unemployment:\nGold: {gold_reactivity:.3f}\nCrypto: {crypto_reactivity:.3f}\nMore Reactive: {more_reactive}\nRatio: {reactivity_ratio:.1f}x', 
                   transform=axes[2,1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('visualizations/recent_period_emphasized_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print recent period statistics
    print(f"\n2023+ PERIOD ANALYSIS ({len(recent_data)} weeks)")
    print("=" * 50)
    print(f"Date range: {recent_data['week_ending_date'].min().date()} to {recent_data['week_ending_date'].max().date()}")
    
    # Calculate correlations for recent period
    recent_corr_icsa_gold = recent_data['ICSA'].corr(recent_data['gold_close_avg'])
    recent_corr_icsa_crypto = recent_data['ICSA'].corr(recent_data['crypto_market_cap'])
    recent_corr_gold_crypto = recent_data['gold_close_avg'].corr(recent_data['crypto_market_cap'])
    
    print(f"\nRecent Period Correlations:")
    print(f"ICSA vs Gold:   {recent_corr_icsa_gold:.4f}")
    print(f"ICSA vs Crypto: {recent_corr_icsa_crypto:.4f}")
    print(f"Gold vs Crypto: {recent_corr_gold_crypto:.4f}")
    
    # Price change correlations
    pct_corr_icsa_gold = recent_data['icsa_pct_change'].corr(recent_data['gold_pct_change'])
    pct_corr_icsa_crypto = recent_data['icsa_pct_change'].corr(recent_data['crypto_pct_change'])
    
    print(f"\nPrice Change Correlations:")
    print(f"ICSA vs Gold % Change:   {pct_corr_icsa_gold:.4f}")
    print(f"ICSA vs Crypto % Change: {pct_corr_icsa_crypto:.4f}")
    
    # Volatility analysis
    print(f"\nVolatility Analysis (2023+):")
    print(f"Gold Average Volatility:   {gold_vol_avg:.2f}%")
    print(f"Crypto Average Volatility: {crypto_vol_avg:.2f}%")
    print(f"Crypto is {vol_ratio:.1f}x more volatile than Gold")
    
    # Reactivity analysis
    print(f"\nReactivity to Unemployment Changes:")
    print(f"Gold Reactivity (avg abs correlation):   {gold_reactivity:.3f}")
    print(f"Crypto Reactivity (avg abs correlation): {crypto_reactivity:.3f}")
    print(f"{more_reactive} is {reactivity_ratio:.1f}x more reactive to unemployment changes")
    
    return recent_data

def create_correlation_analysis(df):
    """Create correlation heatmap and scatter plots"""
    
    # Select key variables for correlation analysis
    corr_vars = ['ICSA', 'gold_close_avg', 'crypto_market_cap', 'crypto_total_volume', 
                 'icsa_pct_change', 'gold_pct_change', 'crypto_pct_change']
    
    corr_df = df[corr_vars].dropna()
    
    # Calculate correlation matrix
    correlation_matrix = corr_df.corr()
    
    # Create correlation heatmap
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Correlation Analysis: Unemployment, Gold, and Cryptocurrency', fontsize=16, fontweight='bold')
    
    # 1. Overall correlation heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=axes[0,0], cbar_kws={'label': 'Correlation Coefficient'})
    axes[0,0].set_title('Correlation Matrix - All Variables', fontweight='bold')
    
    # 2. ICSA vs Gold scatter plot
    axes[0,1].scatter(df['ICSA'], df['gold_close_avg'], alpha=0.6, color='orange')
    z = np.polyfit(df['ICSA'].dropna(), df['gold_close_avg'].dropna(), 1)
    p = np.poly1d(z)
    axes[0,1].plot(df['ICSA'], p(df['ICSA']), "r--", alpha=0.8)
    axes[0,1].set_xlabel('ICSA (Unemployment Claims)')
    axes[0,1].set_ylabel('Gold Price ($)')
    axes[0,1].set_title('Unemployment Claims vs Gold Prices', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    corr_icsa_gold = df['ICSA'].corr(df['gold_close_avg'])
    axes[0,1].text(0.05, 0.95, f'Correlation: {corr_icsa_gold:.3f}', 
                   transform=axes[0,1].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 3. ICSA vs Crypto scatter plot
    axes[1,0].scatter(df['ICSA'], df['crypto_market_cap']/1e12, alpha=0.6, color='blue')
    z = np.polyfit(df['ICSA'].dropna(), (df['crypto_market_cap']/1e12).dropna(), 1)
    p = np.poly1d(z)
    axes[1,0].plot(df['ICSA'], p(df['ICSA']), "r--", alpha=0.8)
    axes[1,0].set_xlabel('ICSA (Unemployment Claims)')
    axes[1,0].set_ylabel('Crypto Market Cap (Trillions $)')
    axes[1,0].set_title('Unemployment Claims vs Crypto Market Cap', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    corr_icsa_crypto = df['ICSA'].corr(df['crypto_market_cap'])
    axes[1,0].text(0.05, 0.95, f'Correlation: {corr_icsa_crypto:.3f}', 
                   transform=axes[1,0].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 4. Gold vs Crypto scatter plot
    axes[1,1].scatter(df['gold_close_avg'], df['crypto_market_cap']/1e12, alpha=0.6, color='purple')
    z = np.polyfit(df['gold_close_avg'].dropna(), (df['crypto_market_cap']/1e12).dropna(), 1)
    p = np.poly1d(z)
    axes[1,1].plot(df['gold_close_avg'], p(df['gold_close_avg']), "r--", alpha=0.8)
    axes[1,1].set_xlabel('Gold Price ($)')
    axes[1,1].set_ylabel('Crypto Market Cap (Trillions $)')
    axes[1,1].set_title('Gold Prices vs Crypto Market Cap', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    corr_gold_crypto = df['gold_close_avg'].corr(df['crypto_market_cap'])
    axes[1,1].text(0.05, 0.95, f'Correlation: {corr_gold_crypto:.3f}', 
                   transform=axes[1,1].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig('visualizations/correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix

def create_volatility_analysis(df):
    """Analyze volatility and percentage changes"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Volatility Analysis: Percentage Changes Over Time', fontsize=16, fontweight='bold')
    
    # 1. Percentage changes over time
    axes[0,0].plot(df['week_ending_date'], df['icsa_pct_change'], alpha=0.7, color='red', label='ICSA')
    axes[0,0].plot(df['week_ending_date'], df['gold_pct_change'], alpha=0.7, color='gold', label='Gold')
    axes[0,0].plot(df['week_ending_date'], df['crypto_pct_change'], alpha=0.7, color='blue', label='Crypto')
    axes[0,0].set_title('Weekly Percentage Changes', fontweight='bold')
    axes[0,0].set_ylabel('% Change')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 2. Distribution of percentage changes
    pct_changes = df[['icsa_pct_change', 'gold_pct_change', 'crypto_pct_change']].dropna()
    pct_changes.boxplot(ax=axes[0,1])
    axes[0,1].set_title('Distribution of Weekly % Changes', fontweight='bold')
    axes[0,1].set_ylabel('% Change')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Rolling volatility (standard deviation)
    window = 12  # 12-week rolling window
    df['icsa_volatility'] = df['icsa_pct_change'].rolling(window=window).std()
    df['gold_volatility'] = df['gold_pct_change'].rolling(window=window).std()
    df['crypto_volatility'] = df['crypto_pct_change'].rolling(window=window).std()
    
    axes[1,0].plot(df['week_ending_date'], df['icsa_volatility'], color='red', label='ICSA Volatility')
    axes[1,0].plot(df['week_ending_date'], df['gold_volatility'], color='gold', label='Gold Volatility')
    axes[1,0].plot(df['week_ending_date'], df['crypto_volatility'], color='blue', label='Crypto Volatility')
    axes[1,0].set_title(f'{window}-Week Rolling Volatility', fontweight='bold')
    axes[1,0].set_ylabel('Volatility (Std Dev of % Changes)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Correlation of percentage changes
    pct_corr = pct_changes.corr()
    sns.heatmap(pct_corr, annot=True, cmap='RdBu_r', center=0, square=True, ax=axes[1,1])
    axes[1,1].set_title('Correlation of Weekly % Changes', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/volatility_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_crisis_analysis(df):
    """Analyze behavior during crisis periods"""
    
    # Define crisis periods
    covid_period = (df['week_ending_date'] >= '2020-03-01') & (df['week_ending_date'] <= '2021-06-01')
    crypto_crash_2018 = (df['week_ending_date'] >= '2018-01-01') & (df['week_ending_date'] <= '2018-12-31')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Crisis Period Analysis', fontsize=16, fontweight='bold')
    
    # 1. COVID-19 period analysis
    covid_data = df[covid_period]
    axes[0,0].plot(covid_data['week_ending_date'], covid_data['ICSA'], color='red', linewidth=2, label='ICSA')
    axes[0,0].set_title('Unemployment Claims During COVID-19', fontweight='bold')
    axes[0,0].set_ylabel('ICSA Claims')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Asset performance during COVID
    ax2 = axes[0,1]
    ax2.plot(covid_data['week_ending_date'], covid_data['gold_close_avg'], color='gold', linewidth=2, label='Gold')
    ax2.set_ylabel('Gold Price ($)', color='gold')
    ax2.tick_params(axis='y', labelcolor='gold')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(covid_data['week_ending_date'], covid_data['crypto_market_cap']/1e12, color='blue', linewidth=2, label='Crypto')
    ax2_twin.set_ylabel('Crypto Market Cap (Trillions $)', color='blue')
    ax2_twin.tick_params(axis='y', labelcolor='blue')
    
    ax2.set_title('Gold vs Crypto During COVID-19', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Average values by year
    yearly_avg = df.groupby('year').agg({
        'ICSA': 'mean',
        'gold_close_avg': 'mean',
        'crypto_market_cap': 'mean'
    }).reset_index()
    
    axes[1,0].bar(yearly_avg['year'], yearly_avg['ICSA'], color='red', alpha=0.7)
    axes[1,0].set_title('Average Annual Unemployment Claims', fontweight='bold')
    axes[1,0].set_ylabel('Average ICSA')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Safe haven analysis - correlation during high unemployment periods
    high_unemployment = df['ICSA'] > df['ICSA'].quantile(0.75)  # Top 25% unemployment periods
    normal_periods = df['ICSA'] <= df['ICSA'].quantile(0.75)
    
    corr_high_unemployment = df[high_unemployment][['ICSA', 'gold_close_avg', 'crypto_market_cap']].corr()
    corr_normal = df[normal_periods][['ICSA', 'gold_close_avg', 'crypto_market_cap']].corr()
    
    # Show correlation difference
    corr_diff = corr_high_unemployment - corr_normal
    sns.heatmap(corr_diff, annot=True, cmap='RdBu_r', center=0, square=True, ax=axes[1,1])
    axes[1,1].set_title('Correlation Difference:\nHigh Unemployment vs Normal Periods', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/crisis_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_statistical_insights(df, correlation_matrix):
    """Generate statistical insights and summary"""
    
    print("\n" + "="*80)
    print("STATISTICAL INSIGHTS: UNEMPLOYMENT, GOLD, AND CRYPTOCURRENCY ANALYSIS")
    print("="*80)
    
    # Basic statistics
    print("\n1. BASIC STATISTICS:")
    print("-" * 40)
    stats_df = df[['ICSA', 'gold_close_avg', 'crypto_market_cap']].describe()
    print(stats_df.round(2))
    
    # Correlation insights
    print("\n2. KEY CORRELATIONS:")
    print("-" * 40)
    icsa_gold_corr = correlation_matrix.loc['ICSA', 'gold_close_avg']
    icsa_crypto_corr = correlation_matrix.loc['ICSA', 'crypto_market_cap']
    gold_crypto_corr = correlation_matrix.loc['gold_close_avg', 'crypto_market_cap']
    
    print(f"ICSA vs Gold Price:        {icsa_gold_corr:.4f}")
    print(f"ICSA vs Crypto Market Cap: {icsa_crypto_corr:.4f}")
    print(f"Gold vs Crypto Market Cap: {gold_crypto_corr:.4f}")
    
    # Interpret correlations
    print("\n3. CORRELATION INTERPRETATION:")
    print("-" * 40)
    
    def interpret_correlation(corr, var1, var2):
        if abs(corr) < 0.1:
            strength = "negligible"
        elif abs(corr) < 0.3:
            strength = "weak"
        elif abs(corr) < 0.5:
            strength = "moderate"
        elif abs(corr) < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if corr > 0 else "negative"
        return f"{strength} {direction}"
    
    print(f"• Unemployment vs Gold: {interpret_correlation(icsa_gold_corr, 'ICSA', 'Gold')} correlation")
    print(f"• Unemployment vs Crypto: {interpret_correlation(icsa_crypto_corr, 'ICSA', 'Crypto')} correlation")
    print(f"• Gold vs Crypto: {interpret_correlation(gold_crypto_corr, 'Gold', 'Crypto')} correlation")
    
    # Volatility analysis
    print("\n4. VOLATILITY ANALYSIS:")
    print("-" * 40)
    icsa_vol = df['icsa_pct_change'].std()
    gold_vol = df['gold_pct_change'].std()
    crypto_vol = df['crypto_pct_change'].std()
    
    print(f"ICSA Weekly Volatility:   {icsa_vol:.2f}%")
    print(f"Gold Weekly Volatility:   {gold_vol:.2f}%")
    print(f"Crypto Weekly Volatility: {crypto_vol:.2f}%")
    
    # COVID impact
    print("\n5. COVID-19 IMPACT ANALYSIS:")
    print("-" * 40)
    covid_period = (df['week_ending_date'] >= '2020-03-01') & (df['week_ending_date'] <= '2021-06-01')
    pre_covid = df['week_ending_date'] < '2020-03-01'
    
    covid_icsa_avg = df[covid_period]['ICSA'].mean()
    pre_covid_icsa_avg = df[pre_covid]['ICSA'].mean()
    icsa_increase = ((covid_icsa_avg - pre_covid_icsa_avg) / pre_covid_icsa_avg) * 100
    
    covid_gold_avg = df[covid_period]['gold_close_avg'].mean()
    pre_covid_gold_avg = df[pre_covid]['gold_close_avg'].mean()
    gold_increase = ((covid_gold_avg - pre_covid_gold_avg) / pre_covid_gold_avg) * 100
    
    covid_crypto_avg = df[covid_period]['crypto_market_cap'].mean()
    pre_covid_crypto_avg = df[pre_covid]['crypto_market_cap'].mean()
    crypto_increase = ((covid_crypto_avg - pre_covid_crypto_avg) / pre_covid_crypto_avg) * 100
    
    print(f"ICSA increase during COVID:   {icsa_increase:.1f}%")
    print(f"Gold price change during COVID: {gold_increase:.1f}%")
    print(f"Crypto market cap change during COVID: {crypto_increase:.1f}%")
    
    # Key insights
    print("\n6. KEY INSIGHTS:")
    print("-" * 40)
    print("• Cryptocurrency shows the highest volatility among all three assets")
    print("• Gold and crypto show positive correlation, suggesting both may serve as alternative investments")
    
    if icsa_gold_corr > 0.1:
        print("• Higher unemployment tends to coincide with higher gold prices (safe haven effect)")
    elif icsa_gold_corr < -0.1:
        print("• Higher unemployment tends to coincide with lower gold prices")
    else:
        print("• Unemployment and gold prices show little direct relationship")
    
    if icsa_crypto_corr > 0.1:
        print("• Higher unemployment coincides with higher crypto market cap")
    elif icsa_crypto_corr < -0.1:
        print("• Higher unemployment coincides with lower crypto market cap")
    else:
        print("• Unemployment and crypto market cap show little direct relationship")
    
    print("\n" + "="*80)

def main():
    """Main function to run all analyses"""
    
    # Create visualizations directory
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    print("\nCreating time series analysis (Linear vs Log)...")
    create_time_series_analysis(df)
    
    print("\nCreating log-scale comparison analysis...")
    log_correlation_matrix = create_log_comparison_analysis(df)
    
    print("\nCreating recent period analysis (2023+)...")
    recent_data = create_recent_period_analysis(df)
    
    print("\nCreating correlation analysis...")
    correlation_matrix = create_correlation_analysis(df)
    
    print("\nCreating volatility analysis...")
    create_volatility_analysis(df)
    
    print("\nCreating crisis period analysis...")
    create_crisis_analysis(df)
    
    print("\nGenerating statistical insights...")
    generate_statistical_insights(df, correlation_matrix)
    
    # Add log-scale insights
    print("\n7. LOG-SCALE CORRELATION INSIGHTS:")
    print("-" * 40)
    log_icsa_gold = log_correlation_matrix.loc['log_icsa', 'log_gold']
    log_icsa_crypto = log_correlation_matrix.loc['log_icsa', 'log_crypto']
    log_gold_crypto = log_correlation_matrix.loc['log_gold', 'log_crypto']
    
    print(f"Log(ICSA) vs Log(Gold):   {log_icsa_gold:.4f}")
    print(f"Log(ICSA) vs Log(Crypto): {log_icsa_crypto:.4f}")
    print(f"Log(Gold) vs Log(Crypto): {log_gold_crypto:.4f}")
    
    print("\n• Log transformation reveals clearer relationships by reducing the impact of extreme values")
    print("• The COVID unemployment spike is normalized, allowing better visualization of underlying trends")
    print("• Log correlations often show stronger relationships than linear correlations for exponential data")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("Visualizations saved in 'visualizations/' directory:")
    print("• time_series_linear_vs_log.png")
    print("• log_scale_analysis.png")
    print("• recent_period_emphasized_analysis.png")
    print("• correlation_analysis.png") 
    print("• volatility_analysis.png")
    print("• crisis_analysis.png")
    print("="*80)

if __name__ == "__main__":
    main()
