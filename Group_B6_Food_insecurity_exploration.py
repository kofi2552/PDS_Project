#!/usr/bin/env python
# coding: utf-8

"""
An Exploratory Analysis of Food Insecurity Patterns in Ghana
Group B6

Group Members:
- Masuda Tuntaya Mashoud (22424730)
- Cephas Amoako Dakwa (22424548)
- Bernard Kofi Ofori Essiamah (22424217)
- Philemon Elikem Kordorwu (22424510)

Date: 31st January, 2026
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- Configuration & Constants ---

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work", "ghana_food_insecurity_clean.csv")

# Variable Mappings
REGION_NAMES = {
    260.0: 'Western', 261.0: 'Central', 262.0: 'Greater Accra',
    263.0: 'Volta', 264.0: 'Eastern', 265.0: 'Ashanti',
    266.0: 'Brong Ahafo', 267.0: 'Northern', 268.0: 'Upper East',
    269.0: 'Upper West', 270.0: 'Ahafo', 271.0: 'Bono',
    272.0: 'Bono East', 273.0: 'Savannah', 274.0: 'North East',
    275.0: 'Oti'
}

FOOD_LABELS = {0: 'Never', 1: 'Once/Twice', 2: 'Several times', 3: 'Many times', 4: 'Always'}
LOCATION_LABELS = {1: 'Urban', 2: 'Rural'}
EDUCATION_MAPPING = {
    0: 'No formal education',
    1: 'Primary', 2: 'Primary', 3: 'Primary',
    4: 'Secondary', 5: 'Secondary', 6: 'Secondary',
    7: 'Tertiary', 8: 'Tertiary', 9: 'Tertiary'
}

ASSET_COLS = ['Q90A', 'Q90B', 'Q90C', 'Q90D', 'Q90E', 'Q90F']
ASSET_NAMES = ['Radio', 'TV', 'Car/Motorcycle', 'Computer', 'Bank Account', 'Mobile Phone']
# Asset columns for score calculation (duplicated in original notebook, consolidated here)
ASSET_SCORE_COLS = ['Q6A', 'Q6B', 'Q6C', 'Q6E', 'Q94', 'Q93A'] # This was in original In[81], checking if it was a mistake. 

DEPRIVATION_VARS = {
    'Q6A': 'Food Shortage',
    'Q6B': 'Water Shortage',
    'Q6C': 'Medical Care Shortage',
    'Q6E': 'Cash Income Shortage'
}

# --- Helper Functions ---

def setup_plotting_style():
    """Configures global plotting settings."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

def load_data(filepath):
    """Loads and returns the dataset."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✓ Data loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def preprocess_data(df):
    """Performs initial data cleaning and feature engineering."""
    # region mapping
    df['region_name'] = df['REGION'].map(REGION_NAMES)
    
    # Asset score (cleaning up the potential confusion from original notebook)
    df['asset_score'] = df[ASSET_COLS].sum(axis=1)
    
    # Education level
    df['education_level'] = df['Q94'].map(EDUCATION_MAPPING)
    
    # Location type
    location_map = {1.0: 'Urban', 2.0: 'Rural'}
    df['location_type'] = df['URBRUR'].map(location_map)
    
    return df

def summarize_categorical(df, column, label_dict=None):
    """Summarizes a categorical variable."""
    if label_dict:
        series = df[column].map(label_dict)
    else:
        series = df[column]

    summary = series.value_counts(dropna=False).sort_index().to_frame()
    summary.columns = ['Count']
    summary['Percentage (%)'] = (summary['Count'] / len(df) * 100).round(2)
    return summary

def analyze_general_stats(df):
    """Prints general summary statistics of key variables."""
    print("\n--- General Statistics ---")
    print("### Food Insecurity Levels (Q6A)")
    print(summarize_categorical(df, 'Q6A', FOOD_LABELS))
    print("\n### Location Type")
    print(summarize_categorical(df, 'URBRUR', LOCATION_LABELS))

def analyze_geography(df):
    """RQ1: Geographic Distribution of Food Insecurity."""
    print("\n--- RQ1: Geographic Distribution ---")
    regional_food = df.groupby('region_name')['is_food_insecure'].agg(['sum', 'count'])
    regional_food['percentage'] = (regional_food['sum'] / regional_food['count']) * 100
    regional_food = regional_food.sort_values('percentage', ascending=False)
    
    print(regional_food)
    
    # Visualization
    plt.figure(figsize=(16, 7))
    regions = regional_food.index.tolist()
    colors = ['#e74c3c' if p > 50 else '#f39c12' if p > 35 else '#27ae60' for p in regional_food['percentage']]
    
    bars = plt.bar(range(len(regions)), regional_food['percentage'], color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
    
    plt.xlabel('Region', fontweight='bold', fontsize=13)
    plt.ylabel('Food Insecurity Rate (%)', fontweight='bold', fontsize=13)
    plt.title('Food Insecurity Rates Across Ghana Regions', fontweight='bold', fontsize=15, pad=20)
    plt.xticks(range(len(regions)), regions, rotation=45, ha='right', fontsize=11)
    
    national_avg = regional_food['percentage'].mean()
    plt.axhline(y=national_avg, color='blue', linestyle='--', linewidth=2, label=f'National Average ({national_avg:.1f}%)')
    
    # Labels
    for bar, val in zip(bars, regional_food['percentage']):
        plt.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.8, edgecolor='black', label='High (>50%)'),
        Patch(facecolor='#f39c12', alpha=0.8, edgecolor='black', label='Medium (35-50%)'),
        Patch(facecolor='#27ae60', alpha=0.8, edgecolor='black', label='Low (<35%)'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='National Average')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.show()

def analyze_assets(df):
    """RQ2: Socio-economic Status and Food Insecurity."""
    print("\n--- RQ2: Asset Ownership & Food Insecurity ---")
    
    # Wealth categories
    df['wealth_category'] = pd.cut(df['asset_score'], bins=[-0.1, 2, 4, 6], labels=['Low (0-2)', 'Medium (3-4)', 'High (5-6)'])
    wealth_food = df.groupby('wealth_category', observed=True)['is_food_insecure'].agg(['sum', 'count'])
    wealth_food['percentage'] = (wealth_food['sum'] / wealth_food['count']) * 100
    print(wealth_food)
    
    # Violin Plot
    plt.figure(figsize=(12, 8))
    parts = plt.violinplot(
        [df[df['is_food_insecure'] == 0]['asset_score'], df[df['is_food_insecure'] == 1]['asset_score']],
        positions=[1, 2], showmeans=True, showmedians=True, widths=0.7
    )
    
    colors = ['#2ecc71', '#e74c3c']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    plt.xticks([1, 2], ['Food Secure', 'Food Insecure'], fontsize=13, fontweight='bold')
    plt.ylabel('Asset Ownership Score (0-6)', fontweight='bold', fontsize=13)
    plt.title('Distribution of Household Assets by Food Security Status', fontweight='bold', fontsize=15)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.show()

def analyze_education(df):
    """RQ3: Education Level and Food Insecurity."""
    print("\n--- RQ3: Education Level ---")
    edu_food = df.groupby('education_level')['is_food_insecure'].agg(['sum', 'count'])
    edu_food['percentage'] = (edu_food['sum'] / edu_food['count']) * 100
    edu_food = edu_food.reindex(['No formal education', 'Primary', 'Secondary', 'Tertiary'])
    
    print(edu_food)
    
    # Line Chart
    plt.figure(figsize=(12, 7))
    x_positions = range(len(edu_food))
    plt.plot(x_positions, edu_food['percentage'], marker='o', markersize=15, linewidth=3, 
             color='#3498db', markerfacecolor='#e74c3c', markeredgecolor='black', label='Food Insecurity Rate')
    
    plt.xticks(x_positions, ['No Formal\nEducation', 'Primary\nSchool', 'Secondary\nSchool', 'Tertiary\n(University)'], fontsize=11)
    plt.ylabel('Food Insecurity Rate (%)', fontweight='bold', fontsize=13)
    plt.title('Impact of Education on Food Insecurity', fontweight='bold', fontsize=15)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

def analyze_deprivation(df):
    """RQ4: Multiple Deprivation and Clustering."""
    print("\n--- RQ4: Multiple Deprivations ---")
    
    # Binary deprivations
    for code, name in DEPRIVATION_VARS.items():
        df[f'{name}_binary'] = (df[code] > 0).astype(int)
    
    deprivation_data = df[['Q6A', 'Q6B', 'Q6C', 'Q6E']].copy()
    deprivation_data.columns = ['Food', 'Water', 'Medical Care', 'Cash Income']
    corr_matrix = deprivation_data.corr()
    
    print("Correlation Matrix:\n", corr_matrix.round(3))
    
    # Heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='YlOrRd', center=0.5, square=True, fmt='.3f')
    plt.title('Correlation Between Deprivations', fontweight='bold', fontsize=15)
    plt.show()
    
    # Deprivation Count Pie Chart
    df['deprivation_count'] = (df[['Q6A', 'Q6B', 'Q6C', 'Q6E']] > 0).sum(axis=1)
    deprivation_dist = df['deprivation_count'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 8))
    labels = [f'{int(i)} Deprivations' for i in deprivation_dist.index]
    plt.pie(deprivation_dist.values, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('Households by Number of Deprivations', fontweight='bold', fontsize=14)
    plt.show()

def analyze_urban_rural(df):
    """RQ5: Urban-Rural Asset Distribution."""
    print("\n--- RQ5: Urban vs Rural Asset Distribution ---")
    
    # Box Plot
    plt.figure(figsize=(14, 9))
    
    urban_secure = df[(df['location_type'] == 'Urban') & (df['is_food_insecure'] == 0)]['asset_score']
    urban_insecure = df[(df['location_type'] == 'Urban') & (df['is_food_insecure'] == 1)]['asset_score']
    rural_secure = df[(df['location_type'] == 'Rural') & (df['is_food_insecure'] == 0)]['asset_score']
    rural_insecure = df[(df['location_type'] == 'Rural') & (df['is_food_insecure'] == 1)]['asset_score']
    
    box_data = [urban_secure, urban_insecure, rural_secure, rural_insecure]
    plt.boxplot(box_data, positions=[1, 2, 4, 5], patch_artist=True, showmeans=True)
    
    plt.xticks([1.5, 4.5], ['URBAN', 'RURAL'], fontsize=13, fontweight='bold')
    plt.ylabel('Asset Ownership Score (0-6)', fontweight='bold', fontsize=13)
    plt.title('Urban vs Rural Asset Ownership by Food Security', fontweight='bold', fontsize=16)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze Food Insecurity Data")
    parser.add_argument("filepath", nargs="?", default=DEFAULT_DATA_PATH, help="Path to the cleaned CSV file")
    parser.add_argument("--no-show", action="store_true", help="Do not display plots (for batch/test mode)")
    args = parser.parse_args()
    
    setup_plotting_style()
    df = load_data(args.filepath)
    
    if df is not None:
        df = preprocess_data(df)
        
        analyze_general_stats(df)
        analyze_geography(df)
        if not args.no_show: plt.show()
        else: plt.close() # Close previous plot to avoid memory warnings if loop
        
        analyze_assets(df)
        if not args.no_show: plt.show()
        else: plt.close()

        analyze_education(df)
        if not args.no_show: plt.show()
        else: plt.close()

        analyze_deprivation(df)
        if not args.no_show: plt.show()
        else: plt.close()

        analyze_urban_rural(df)
        if not args.no_show: plt.show()
        else: plt.close()
        
        print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
