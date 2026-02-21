import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Import functions and constants from the original script
from Group_B6_Food_insecurity_exploration import (
    load_data, preprocess_data, 
    setup_plotting_style, summarize_categorical, FOOD_LABELS, LOCATION_LABELS, DEPRIVATION_VARS
)

# Use a relative path assuming you are running from the project root
DATA_PATH = "work/ghana_food_insecurity_clean.csv"

# Streamlit Page Config
st.set_page_config(page_title="Ghana Food Insecurity Dashboard", page_icon="🌾", layout="wide")

st.title("🌾 Ghana Food Insecurity Dashboard")
st.markdown("""
An Exploratory Analysis of Food Insecurity Patterns in Ghana (Group B6). 
Explore geographic distribution, socio-economic status, education levels and more using the interactive dashboard below.
""")

setup_plotting_style()

@st.cache_data
def get_data():
    df = load_data(DATA_PATH)
    if df is not None:
        df = preprocess_data(df)
    return df

with st.spinner("Loading data..."):
    df = get_data()

if df is None:
    st.error(f"Failed to load data from `{DATA_PATH}`. Please check the data path.")
    st.stop()

# --- Overview ---
st.header("1. General Statistics")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Food Insecurity Levels (Q6A)")
    food_stats = summarize_categorical(df, 'Q6A', FOOD_LABELS)
    st.dataframe(food_stats, use_container_width=True)

with col2:
    st.subheader("Location Type Distribution")
    loc_stats = summarize_categorical(df, 'URBRUR', LOCATION_LABELS)
    st.dataframe(loc_stats, use_container_width=True)

st.divider()

# --- Geographic Distribution ---
st.header("2. Geographic Distribution of Food Insecurity")
regional_food = df.groupby('region_name')['is_food_insecure'].agg(['sum', 'count'])
regional_food['percentage'] = (regional_food['sum'] / regional_food['count']) * 100
regional_food = regional_food.sort_values('percentage', ascending=False)

fig1, ax1 = plt.subplots(figsize=(16, 7))
regions = regional_food.index.tolist()
colors = ['#e74c3c' if p > 50 else '#f39c12' if p > 35 else '#27ae60' for p in regional_food['percentage']]
bars = ax1.bar(range(len(regions)), regional_food['percentage'], color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)

ax1.set_xlabel('Region', fontweight='bold', fontsize=13)
ax1.set_ylabel('Food Insecurity Rate (%)', fontweight='bold', fontsize=13)
ax1.set_title('Food Insecurity Rates Across Ghana Regions', fontweight='bold', fontsize=15, pad=20)
ax1.set_xticks(range(len(regions)))
ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=11)

national_avg = regional_food['percentage'].mean()
ax1.axhline(y=national_avg, color='blue', linestyle='--', linewidth=2, label=f'National Average ({national_avg:.1f}%)')

for bar, val in zip(bars, regional_food['percentage']):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

legend_elements = [
    Patch(facecolor='#e74c3c', alpha=0.8, edgecolor='black', label='High (>50%)'),
    Patch(facecolor='#f39c12', alpha=0.8, edgecolor='black', label='Medium (35-50%)'),
    Patch(facecolor='#27ae60', alpha=0.8, edgecolor='black', label='Low (<35%)'),
    Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='National Average')
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
ax1.set_ylim(0, 105)
plt.tight_layout()
st.pyplot(fig1)

st.divider()

# --- Asset Ownership ---
st.header("3. Asset Ownership & Food Insecurity")
st.markdown("Distribution of Household Assets by Food Security Status.")

fig2, ax2 = plt.subplots(figsize=(12, 8))
# Filter out nulls if any
fs_assets = df[df['is_food_insecure'] == 0]['asset_score'].dropna()
fi_assets = df[df['is_food_insecure'] == 1]['asset_score'].dropna()

parts = ax2.violinplot(
    [fs_assets, fi_assets],
    positions=[1, 2], showmeans=True, showmedians=True, widths=0.7
)

colors2 = ['#2ecc71', '#e74c3c']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors2[i])
    pc.set_alpha(0.7)

ax2.set_xticks([1, 2])
ax2.set_xticklabels(['Food Secure', 'Food Insecure'], fontsize=13, fontweight='bold')
ax2.set_ylabel('Asset Ownership Score (0-6)', fontweight='bold', fontsize=13)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
st.pyplot(fig2)

st.divider()

# --- Education Level ---
st.header("4. Education Level and Food Insecurity")

edu_food = df.groupby('education_level')['is_food_insecure'].agg(['sum', 'count'])
edu_food['percentage'] = (edu_food['sum'] / edu_food['count']) * 100
edu_food = edu_food.reindex(['No formal education', 'Primary', 'Secondary', 'Tertiary'])

fig3, ax3 = plt.subplots(figsize=(12, 7))
x_positions = range(len(edu_food))
ax3.plot(x_positions, edu_food['percentage'], marker='o', markersize=15, linewidth=3, 
         color='#3498db', markerfacecolor='#e74c3c', markeredgecolor='black', label='Food Insecurity Rate')

ax3.set_xticks(x_positions)
ax3.set_xticklabels(['No Formal\nEducation', 'Primary\nSchool', 'Secondary\nSchool', 'Tertiary\n(University)'], fontsize=11)
ax3.set_ylabel('Food Insecurity Rate (%)', fontweight='bold', fontsize=13)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
st.pyplot(fig3)

st.divider()

# --- Deprivations ---
st.header("5. Multiple Deprivations")
col3, col4 = st.columns(2)

for code, name in DEPRIVATION_VARS.items():
    df[f'{name}_binary'] = (df[code] > 0).astype(int)

deprivation_data = df[['Q6A', 'Q6B', 'Q6C', 'Q6E']].copy()
deprivation_data.columns = ['Food', 'Water', 'Medical Care', 'Cash Income']
corr_matrix = deprivation_data.corr()

with col3:
    st.subheader("Correlation Between Deprivations")
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='YlOrRd', center=0.5, square=True, fmt='.3f', ax=ax4)
    st.pyplot(fig4)

with col4:
    st.subheader("Households by Number of Deprivations")
    df['deprivation_count'] = (df[['Q6A', 'Q6B', 'Q6C', 'Q6E']] > 0).sum(axis=1)
    deprivation_dist = df['deprivation_count'].value_counts().sort_index()
    
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    labels = [f'{int(i)} Deprivations' for i in deprivation_dist.index]
    ax5.pie(deprivation_dist.values, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
    st.pyplot(fig5)

st.divider()

# --- Urban vs Rural ---
st.header("6. Urban vs Rural Asset Distribution")
fig6, ax6 = plt.subplots(figsize=(14, 9))

urban_secure = df[(df['location_type'] == 'Urban') & (df['is_food_insecure'] == 0)]['asset_score'].dropna()
urban_insecure = df[(df['location_type'] == 'Urban') & (df['is_food_insecure'] == 1)]['asset_score'].dropna()
rural_secure = df[(df['location_type'] == 'Rural') & (df['is_food_insecure'] == 0)]['asset_score'].dropna()
rural_insecure = df[(df['location_type'] == 'Rural') & (df['is_food_insecure'] == 1)]['asset_score'].dropna()

box_data = [urban_secure, urban_insecure, rural_secure, rural_insecure]
ax6.boxplot(box_data, positions=[1, 2, 4, 5], patch_artist=True, showmeans=True)

ax6.set_xticks([1.5, 4.5])
ax6.set_xticklabels(['URBAN', 'RURAL'], fontsize=13, fontweight='bold')
ax6.set_ylabel('Asset Ownership Score (0-6)', fontweight='bold', fontsize=13)
st.pyplot(fig6)
