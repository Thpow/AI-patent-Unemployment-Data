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
    """Load the no-COVID dataset and calculate week-over-week percentage changes"""
    
    # Load the overlap-only dataset without COVID
    data_file = "3_combined_data/final_data_overlap_only_no_covid.csv"
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run combine_data_no_covid.py first.")
        return None
    
    df = pd.read_csv(data_file)
    df['week_ending_date'] = pd.to_datetime(df['week_ending_date'])
    
    print(f"Loaded dataset: {len(df)} weeks from {df['week_ending_date'].min()} to {df['week_ending_date'].max()}")
    print(f"No COVID period data (2020-2021 excluded)")
    
    # Calculate week-over-week percentage changes
    df['icsa_pct_change'] = df['ICSA'].pct_change() * 100
    df['gold_pct_change'] = df['gold_close_avg'].pct_change() * 100
    df['crypto_pct_change'] = df['crypto_market_cap'].pct_change() * 100
    
    # Remove first row (NaN values from pct_change)
    df = df.dropna(subset=['icsa_pct_change', 'gold_pct_change', 'crypto_pct_change'])
    
    print(f"Data after calculating percentage changes: {len(df)} weeks")
    
    return df

def create_weekly_change_comparison(df):
    """Create comprehensive week-over-week percentage change analysis"""
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle('Week-over-Week Percentage Change Analysis (No COVID Period)', fontsize=16, fontweight='bold')
    
    # 1. Time series of percentage changes
    axes[0,0].plot(df['week_ending_date'], df['icsa_pct_change'], 
                   label='ICSA % Change', color='red', alpha=0.7, linewidth=1)
    axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0,0].set_title('ICSA Week-over-Week % Change', fontweight='bold')
    axes[0,0].set_ylabel('Percentage Change (%)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Add volatility bands (±1 std dev)
    icsa_mean = df['icsa_pct_change'].mean()
    icsa_std = df['icsa_pct_change'].std()
    axes[0,0].axhline(y=icsa_mean + icsa_std, color='red', linestyle='--', alpha=0.5, label=f'±1σ ({icsa_std:.1f}%)')
    axes[0,0].axhline(y=icsa_mean - icsa_std, color='red', linestyle='--', alpha=0.5)
    axes[0,0].legend()
    
    # 2. Gold percentage changes
    axes[0,1].plot(df['week_ending_date'], df['gold_pct_change'], 
                   label='Gold % Change', color='gold', alpha=0.7, linewidth=1)
    axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0,1].set_title('Gold Week-over-Week % Change', fontweight='bold')
    axes[0,1].set_ylabel('Percentage Change (%)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Add volatility bands
    gold_mean = df['gold_pct_change'].mean()
    gold_std = df['gold_pct_change'].std()
    axes[0,1].axhline(y=gold_mean + gold_std, color='gold', linestyle='--', alpha=0.5, label=f'±1σ ({gold_std:.1f}%)')
    axes[0,1].axhline(y=gold_mean - gold_std, color='gold', linestyle='--', alpha=0.5)
    axes[0,1].legend()
    
    # 3. Crypto percentage changes
    axes[1,0].plot(df['week_ending_date'], df['crypto_pct_change'], 
                   label='Crypto % Change', color='purple', alpha=0.7, linewidth=1)
    axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1,0].set_title('Crypto Week-over-Week % Change', fontweight='bold')
    axes[1,0].set_ylabel('Percentage Change (%)')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Add volatility bands
    crypto_mean = df['crypto_pct_change'].mean()
    crypto_std = df['crypto_pct_change'].std()
    axes[1,0].axhline(y=crypto_mean + crypto_std, color='purple', linestyle='--', alpha=0.5, label=f'±1σ ({crypto_std:.1f}%)')
    axes[1,0].axhline(y=crypto_mean - crypto_std, color='purple', linestyle='--', alpha=0.5)
    axes[1,0].legend()
    
    # 4. Overlaid comparison (normalized to same scale for visibility)
    # Normalize by dividing by standard deviation to compare relative movements
    icsa_norm = df['icsa_pct_change'] / df['icsa_pct_change'].std()
    gold_norm = df['gold_pct_change'] / df['gold_pct_change'].std()
    crypto_norm = df['crypto_pct_change'] / df['crypto_pct_change'].std()
    
    axes[1,1].plot(df['week_ending_date'], icsa_norm, 
                   label='ICSA (normalized)', color='red', alpha=0.8, linewidth=1.5)
    axes[1,1].plot(df['week_ending_date'], gold_norm, 
                   label='Gold (normalized)', color='gold', alpha=0.8, linewidth=1.5)
    axes[1,1].plot(df['week_ending_date'], crypto_norm, 
                   label='Crypto (normalized)', color='purple', alpha=0.8, linewidth=1.5)
    axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1,1].set_title('Normalized % Changes Comparison', fontweight='bold')
    axes[1,1].set_ylabel('Standard Deviations from Mean')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # 5. Distribution comparison
    axes[2,0].hist(df['icsa_pct_change'], bins=50, alpha=0.6, label='ICSA', color='red', density=True)
    axes[2,0].hist(df['gold_pct_change'], bins=50, alpha=0.6, label='Gold', color='gold', density=True)
    axes[2,0].hist(df['crypto_pct_change'], bins=50, alpha=0.6, label='Crypto', color='purple', density=True)
    axes[2,0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[2,0].set_title('Distribution of % Changes', fontweight='bold')
    axes[2,0].set_xlabel('Percentage Change (%)')
    axes[2,0].set_ylabel('Density')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)
    
    # 6. Volatility comparison bar chart
    volatilities = [df['icsa_pct_change'].std(), df['gold_pct_change'].std(), df['crypto_pct_change'].std()]
    assets = ['ICSA', 'Gold', 'Crypto']
    colors = ['red', 'gold', 'purple']
    
    bars = axes[2,1].bar(assets, volatilities, color=colors, alpha=0.7)
    axes[2,1].set_title('Volatility Comparison (Standard Deviation)', fontweight='bold')
    axes[2,1].set_ylabel('Standard Deviation (%)')
    axes[2,1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, vol in zip(bars, volatilities):
        height = bar.get_height()
        axes[2,1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{vol:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/weekly_change_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_reactivity_analysis(df):
    """Analyze how assets react to unemployment news (changes)"""
    
    # Define significant unemployment changes (above 1 standard deviation)
    icsa_std = df['icsa_pct_change'].std()
    significant_increases = df['icsa_pct_change'] > icsa_std
    significant_decreases = df['icsa_pct_change'] < -icsa_std
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Asset Reactivity to Unemployment News (No COVID Period)', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: ICSA change vs Gold change
    axes[0,0].scatter(df['icsa_pct_change'], df['gold_pct_change'], 
                      alpha=0.6, color='orange', s=30)
    
    # Highlight significant unemployment changes
    axes[0,0].scatter(df.loc[significant_increases, 'icsa_pct_change'], 
                      df.loc[significant_increases, 'gold_pct_change'], 
                      color='red', s=50, alpha=0.8, label='Significant ICSA Increase')
    axes[0,0].scatter(df.loc[significant_decreases, 'icsa_pct_change'], 
                      df.loc[significant_decreases, 'gold_pct_change'], 
                      color='green', s=50, alpha=0.8, label='Significant ICSA Decrease')
    
    # Add trend line
    z = np.polyfit(df['icsa_pct_change'], df['gold_pct_change'], 1)
    p = np.poly1d(z)
    axes[0,0].plot(df['icsa_pct_change'], p(df['icsa_pct_change']), "r--", alpha=0.8, linewidth=2)
    
    axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0,0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[0,0].set_xlabel('ICSA % Change')
    axes[0,0].set_ylabel('Gold % Change')
    axes[0,0].set_title('Gold Reactivity to Unemployment Changes', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    gold_reactivity = df['icsa_pct_change'].corr(df['gold_pct_change'])
    axes[0,0].text(0.05, 0.95, f'Reactivity: {gold_reactivity:.3f}', 
                   transform=axes[0,0].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 2. Scatter plot: ICSA change vs Crypto change
    axes[0,1].scatter(df['icsa_pct_change'], df['crypto_pct_change'], 
                      alpha=0.6, color='lightpurple', s=30)
    
    # Highlight significant unemployment changes
    axes[0,1].scatter(df.loc[significant_increases, 'icsa_pct_change'], 
                      df.loc[significant_increases, 'crypto_pct_change'], 
                      color='red', s=50, alpha=0.8, label='Significant ICSA Increase')
    axes[0,1].scatter(df.loc[significant_decreases, 'icsa_pct_change'], 
                      df.loc[significant_decreases, 'crypto_pct_change'], 
                      color='green', s=50, alpha=0.8, label='Significant ICSA Decrease')
    
    # Add trend line
    z = np.polyfit(df['icsa_pct_change'], df['crypto_pct_change'], 1)
    p = np.poly1d(z)
    axes[0,1].plot(df['icsa_pct_change'], p(df['icsa_pct_change']), "r--", alpha=0.8, linewidth=2)
    
    axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0,1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[0,1].set_xlabel('ICSA % Change')
    axes[0,1].set_ylabel('Crypto % Change')
    axes[0,1].set_title('Crypto Reactivity to Unemployment Changes', fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    crypto_reactivity = df['icsa_pct_change'].corr(df['crypto_pct_change'])
    axes[0,1].text(0.05, 0.95, f'Reactivity: {crypto_reactivity:.3f}', 
                   transform=axes[0,1].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 3. Average reaction to significant unemployment changes
    sig_increase_data = df[significant_increases]
    sig_decrease_data = df[significant_decreases]
    
    reactions = {
        'Gold Reaction to\nICSA Increase': sig_increase_data['gold_pct_change'].mean(),
        'Gold Reaction to\nICSA Decrease': sig_decrease_data['gold_pct_change'].mean(),
        'Crypto Reaction to\nICSA Increase': sig_increase_data['crypto_pct_change'].mean(),
        'Crypto Reaction to\nICSA Decrease': sig_decrease_data['crypto_pct_change'].mean()
    }
    
    labels = list(reactions.keys())
    values = list(reactions.values())
    colors = ['gold', 'gold', 'purple', 'purple']
    
    bars = axes[1,0].bar(labels, values, color=colors, alpha=0.7)
    axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1,0].set_title('Average Asset Reaction to Significant ICSA Changes', fontweight='bold')
    axes[1,0].set_ylabel('Average % Change')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., 
                       height + 0.05 if height >= 0 else height - 0.1,
                       f'{value:.2f}%', ha='center', 
                       va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # 4. Reactivity comparison summary
    reactivity_data = {
        'Asset': ['Gold', 'Crypto'],
        'Reactivity to ICSA': [gold_reactivity, crypto_reactivity],
        'Volatility (%)': [df['gold_pct_change'].std(), df['crypto_pct_change'].std()]
    }
    
    x = np.arange(len(reactivity_data['Asset']))
    width = 0.35
    
    bars1 = axes[1,1].bar(x - width/2, [abs(r) for r in reactivity_data['Reactivity to ICSA']], 
                          width, label='Reactivity (abs)', color=['gold', 'purple'], alpha=0.7)
    bars2 = axes[1,1].bar(x + width/2, [v/50 for v in reactivity_data['Volatility (%)']], 
                          width, label='Volatility (/50)', color=['orange', 'indigo'], alpha=0.7)
    
    axes[1,1].set_title('Reactivity vs Volatility Comparison', fontweight='bold')
    axes[1,1].set_ylabel('Normalized Values')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(reactivity_data['Asset'])
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        axes[1,1].text(bar1.get_x() + bar1.get_width()/2., height1 + 0.01,
                       f'{reactivity_data["Reactivity to ICSA"][i]:.3f}', 
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
        axes[1,1].text(bar2.get_x() + bar2.get_width()/2., height2 + 0.01,
                       f'{reactivity_data["Volatility (%)"][i]:.1f}%', 
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizations/asset_reactivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Asset Reactivity Summary ===")
    print(f"Significant ICSA increases (>{icsa_std:.1f}%): {significant_increases.sum()} weeks")
    print(f"Significant ICSA decreases (<-{icsa_std:.1f}%): {significant_decreases.sum()} weeks")
    print(f"\nReactivity to ICSA changes:")
    print(f"  Gold: {gold_reactivity:.3f}")
    print(f"  Crypto: {crypto_reactivity:.3f}")
    print(f"\nAverage reactions to significant ICSA changes:")
    for label, value in reactions.items():
        print(f"  {label}: {value:.2f}%")

def create_rolling_reactivity_analysis(df):
    """Analyze how reactivity changes over time"""
    
    window = 52  # 1-year rolling window
    
    # Calculate rolling correlations (reactivity)
    rolling_gold_reactivity = df['icsa_pct_change'].rolling(window).corr(df['gold_pct_change'])
    rolling_crypto_reactivity = df['icsa_pct_change'].rolling(window).corr(df['crypto_pct_change'])
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('Rolling Asset Reactivity to Unemployment Changes (52-week window)', fontsize=16, fontweight='bold')
    
    # 1. Rolling reactivity over time
    axes[0].plot(df['week_ending_date'], rolling_gold_reactivity, 
                 label='Gold Reactivity', color='gold', linewidth=2, alpha=0.8)
    axes[0].plot(df['week_ending_date'], rolling_crypto_reactivity, 
                 label='Crypto Reactivity', color='purple', linewidth=2, alpha=0.8)
    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0].set_title('Rolling Reactivity to ICSA Changes', fontweight='bold')
    axes[0].set_ylabel('Correlation (Reactivity)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Reactivity difference (Gold vs Crypto)
    reactivity_diff = rolling_gold_reactivity - rolling_crypto_reactivity
    axes[1].plot(df['week_ending_date'], reactivity_diff, 
                 color='darkgreen', linewidth=2, alpha=0.8)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1].fill_between(df['week_ending_date'], reactivity_diff, 0, 
                         where=(reactivity_diff > 0), color='gold', alpha=0.3, label='Gold More Reactive')
    axes[1].fill_between(df['week_ending_date'], reactivity_diff, 0, 
                         where=(reactivity_diff < 0), color='purple', alpha=0.3, label='Crypto More Reactive')
    axes[1].set_title('Reactivity Difference (Gold - Crypto)', fontweight='bold')
    axes[1].set_ylabel('Reactivity Difference')
    axes[1].set_xlabel('Date')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/rolling_reactivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function for weekly change comparison"""
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    print("=== Week-over-Week Change Analysis (No COVID Period) ===")
    
    # Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Print basic statistics
    print(f"\n=== Weekly Change Statistics ===")
    print(f"ICSA % Change - Mean: {df['icsa_pct_change'].mean():.2f}%, Std: {df['icsa_pct_change'].std():.2f}%")
    print(f"Gold % Change - Mean: {df['gold_pct_change'].mean():.2f}%, Std: {df['gold_pct_change'].std():.2f}%")
    print(f"Crypto % Change - Mean: {df['crypto_pct_change'].mean():.2f}%, Std: {df['crypto_pct_change'].std():.2f}%")
    
    # Create all visualizations
    print("\n1. Creating weekly change comparison...")
    create_weekly_change_comparison(df)
    
    print("\n2. Creating reactivity analysis...")
    create_reactivity_analysis(df)
    
    print("\n3. Creating rolling reactivity analysis...")
    create_rolling_reactivity_analysis(df)
    
    print("\n=== Analysis Complete ===")
    print("All visualizations saved to visualizations/ directory:")
    print("- weekly_change_comparison.png")
    print("- asset_reactivity_analysis.png") 
    print("- rolling_reactivity_analysis.png")
    
    return df

if __name__ == "__main__":
    df = main()
