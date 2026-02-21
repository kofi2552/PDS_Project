import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import functions and constants from the original script
try:
    from Group_B6_Food_insecurity_exploration import (
        load_data, preprocess_data, 
        summarize_categorical, FOOD_LABELS, LOCATION_LABELS, DEPRIVATION_VARS
    )
except ImportError:
    st.error("Could not import functions from Group_B6_Food_insecurity_exploration.py. Ensure the file is in the same directory.")
    st.stop()

# Use a relative path assuming you are running from the project root
DATA_PATH = "work/ghana_food_insecurity_clean.csv"

# Streamlit Page Config
st.set_page_config(page_title="Ghana Food Insecurity Dashboard", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #F9FAFB;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
        text-align: center;
        border-top: 4px solid #3B82F6;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1F2937;
    }
    .metric-delta {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    .delta-positive { color: #10B981; }
    .delta-negative { color: #EF4444; }
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    df = load_data(DATA_PATH)
    if df is not None:
        df = preprocess_data(df)
    return df

with st.spinner("Loading insights..."):
    original_df = get_data()

if original_df is None:
    st.error(f"Failed to load data from `{DATA_PATH}`. Please check the data path.")
    st.stop()

df = original_df.copy()

# --- SIDEBAR INTERACTIVITY & MODELING ---
st.sidebar.markdown('<h2 style="color: #1E3A8A; font-weight: bold;">🌾 Controls & Modeling</h2>', unsafe_allow_html=True)
st.sidebar.markdown("Use these controls to filter the historical data or model hypothetical scenarios.")

st.sidebar.markdown("### 🔍 Filters")
# Region Filter
all_regions = ["All Regions"] + sorted(df['region_name'].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("Filter by Region", all_regions)

if selected_region != "All Regions":
    df = df[df['region_name'] == selected_region]

# Location Filter
location_options = ["Both", "Urban", "Rural"]
selected_location = st.sidebar.radio("Filter Location", location_options, index=0, horizontal=True)

if selected_location != "Both":
    df = df[df['location_type'] == selected_location]

# --- 'What-If' Modeling ---
st.sidebar.markdown("---")
st.sidebar.markdown('### 🔬 "What-If" Modeling')
st.sidebar.info("Adjust the sliders below to simulate policy changes and see how they might impact food security rates based on our dataset trends.")

# Education Simulation
st.sidebar.markdown("**Education Policy Simulation**")
edu_boost = st.sidebar.slider(
    "Convert households with 'No Formal Education' to 'Primary'?",
    min_value=0, max_value=100, value=0, step=10, format="%d%%"
)

# Asset Simulation
st.sidebar.markdown("**Economic Relief Simulation**")
asset_boost = st.sidebar.slider(
    "Increase Household Asset Scores by:",
    min_value=0, max_value=3, value=0, step=1, 
    help="Simulates giving households cash or assets. An increase of +1 assumes they gain one more basic asset category."
)

# Apply Modeling Logic
baseline_insecure_rate = (df['is_food_insecure'].sum() / len(df)) * 100 if len(df) > 0 else 0

if edu_boost > 0 or asset_boost > 0:
    # We create a simulated dataframe for the 'What-If' scenario
    sim_df = df.copy()
    
    # Apply Education Boost Simulation
    if edu_boost > 0:
        # Find indices of households with no formal education
        no_edu_idx = sim_df[sim_df['education_level'] == 'No formal education'].index
        # Calculate how many to "upgrade"
        upgrade_count = int(len(no_edu_idx) * (edu_boost / 100.0))
        if upgrade_count > 0:
            # Randomly select households to upgrade to Primary
            np.random.seed(42) # For consistent results
            upgrade_idx = np.random.choice(no_edu_idx, size=upgrade_count, replace=False)
            sim_df.loc[upgrade_idx, 'education_level'] = 'Primary'
            
            # Simple heuristic model: Upgrading to primary reduces food insecurity log-odds by a factor
            # Based on the data, primary is slightly more secure than no education. We'll simulate a 15% improvement chance.
            currently_insecure_but_upgraded = [idx for idx in upgrade_idx if sim_df.loc[idx, 'is_food_insecure'] == 1]
            fix_count = int(len(currently_insecure_but_upgraded) * 0.15) 
            if fix_count > 0:
                fixed_idx = np.random.choice(currently_insecure_but_upgraded, size=fix_count, replace=False)
                sim_df.loc[fixed_idx, 'is_food_insecure'] = 0
                sim_df.loc[fixed_idx, 'Q6A'] = 0 # 'Never' insecure
    
    # Apply Asset Boost Simulation
    if asset_boost > 0:
        sim_df['asset_score'] = sim_df['asset_score'] + asset_boost
        # Cap asset score at max (6)
        sim_df['asset_score'] = sim_df['asset_score'].clip(upper=6)
        
        # Simple heuristic model: For every asset point gained, 10% chance to become food secure
        insecure_idx = sim_df[sim_df['is_food_insecure'] == 1].index
        fix_prob = min(asset_boost * 0.10, 1.0)
        fix_count = int(len(insecure_idx) * fix_prob)
        
        if fix_count > 0:
            fixed_idx = np.random.choice(insecure_idx, size=fix_count, replace=False)
            sim_df.loc[fixed_idx, 'is_food_insecure'] = 0
            sim_df.loc[fixed_idx, 'Q6A'] = 0

    # Overwrite visualization df with simulation
    df = sim_df


# --- MAIN UI ---
st.markdown('<div class="main-header">🌾 Ghana Food Insecurity Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Exploratory Analysis (Group B6)</div>', unsafe_allow_html=True)

if len(df) == 0:
    st.warning("No data available for the selected filters.")
    st.stop()

# --- Top Level Metrics ---
total_households = len(df)
overall_insecure = df['is_food_insecure'].sum()
insecure_rate = (overall_insecure / total_households) * 100
urban_pct = (df['location_type'] == 'Urban').mean() * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f'<div class="card"><div class="metric-value">{total_households:,}</div><div class="metric-label">Households in View</div></div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="card"><div class="metric-value">{overall_insecure:,}</div><div class="metric-label">Food Insecure</div></div>', unsafe_allow_html=True)

with col3:
    if edu_boost > 0 or asset_boost > 0:
        diff = baseline_insecure_rate - insecure_rate
        color_class = "delta-positive" if diff > 0 else "delta-negative"
        sign = "-" if diff > 0 else "+"
        delta_html = f'<div class="metric-delta {color_class}">({sign}{abs(diff):.1f}% vs baseline)</div>'
        st.markdown(f'<div class="card"><div class="metric-value" style="color:#10B981">{insecure_rate:.1f}%</div>{delta_html}<div class="metric-label">Simulated Insecurity Rate</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="card"><div class="metric-value">{insecure_rate:.1f}%</div><div class="metric-label">Insecurity Rate</div></div>', unsafe_allow_html=True)

with col4:
    st.markdown(f'<div class="card"><div class="metric-value">{urban_pct:.1f}%</div><div class="metric-label">Urban representation</div></div>', unsafe_allow_html=True)


st.write("") # Spacer

# --- Dashboard Content in Tabs ---
tab1, tab2, tab3 = st.tabs(["🌎 Geography & Location", "💰 Wealth & Education", "⚠️ Multiple Deprivations"])

with tab1:
    st.markdown("### Geographic & Location Distribution")
    colA, colB = st.columns([2, 1])
    
    with colA:
        regional_food = df.groupby('region_name')['is_food_insecure'].agg(['sum', 'count']).reset_index()
        regional_food['percentage'] = (regional_food['sum'] / regional_food['count']) * 100
        regional_food = regional_food.sort_values('percentage', ascending=False)
        national_avg = regional_food['percentage'].mean()
        
        # Color coding: red >50, orange 35-50, green <35
        regional_food['Risk Level'] = pd.cut(regional_food['percentage'], 
                                              bins=[-1, 35, 50.001, 101], 
                                              labels=['Low (<35%)', 'Medium (35-50%)', 'High (>50%)'])
        
        color_discrete_map = {'Low (<35%)': '#2ecc71', 'Medium (35-50%)': '#f39c12', 'High (>50%)': '#e74c3c'}
        
        fig1 = px.bar(regional_food, x='region_name', y='percentage', color='Risk Level',
                      title='Food Insecurity Rates Across Ghana Regions',
                      labels={'region_name': 'Region', 'percentage': 'Food Insecurity Rate (%)'},
                      color_discrete_map=color_discrete_map, height=450)
        
        fig1.add_hline(y=national_avg, line_dash="dash", line_color="#3B82F6", annotation_text=f" Group Avg: {national_avg:.1f}% ", annotation_position="top right")
        fig1.update_layout(xaxis_tickangle=-45, margin=dict(t=50, b=0, l=0, r=0), plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        loc_stats = df['location_type'].value_counts().reset_index()
        loc_stats.columns = ['Location', 'Count']
        fig1b = px.pie(loc_stats, values='Count', names='Location', 
                       title='Sample Distribution by Location', hole=0.45,
                       color='Location', color_discrete_map={'Urban': '#3B82F6', 'Rural': '#10B981'}, height=400)
        fig1b.update_layout(margin=dict(t=50, b=0, l=0, r=0))
        st.plotly_chart(fig1b, use_container_width=True)
        
        with st.expander("📝 View Underlying Data Grid"):
            st.dataframe(regional_food[['region_name', 'count', 'percentage']].rename(columns={'region_name':'Region', 'count': 'Sample Size', 'percentage':'Insecurity %'}).round(1), use_container_width=True, hide_index=True)


with tab2:
    st.markdown("### Socio-Economic Factors")
    colC, colD = st.columns(2)
    
    with colC:
        fig2 = px.violin(df.dropna(subset=['asset_score']), y='asset_score', color='is_food_insecure',
                         box=True, points=False,
                         title='Household Assets vs. Food Security',
                         labels={'asset_score': 'Asset Score (0-6)', 'is_food_insecure': 'Food Insecure'},
                         color_discrete_map={0: '#10B981', 1: '#EF4444'}, height=400)
        
        fig2.for_each_trace(lambda t: t.update(name = 'Secure' if t.name == '0' else 'Insecure'))
        fig2.update_layout(margin=dict(t=50, b=0, l=0, r=0), plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Number of Assets")
        st.plotly_chart(fig2, use_container_width=True)
        
    with colD:
        edu_food = df.groupby('education_level')['is_food_insecure'].agg(['sum', 'count']).reset_index()
        edu_food['percentage'] = (edu_food['sum'] / edu_food['count']) * 100
        
        # Sort categorically
        edu_order = ['No formal education', 'Primary', 'Secondary', 'Tertiary']
        # Filter purely string/valid education levels to avoid categorical sorting bugs on None
        valid_edu_food = edu_food[edu_food['education_level'].isin(edu_order)].copy()
        valid_edu_food['education_level'] = pd.Categorical(valid_edu_food['education_level'], categories=edu_order, ordered=True)
        valid_edu_food = valid_edu_food.sort_values('education_level')
        
        fig3 = px.line(valid_edu_food, x='education_level', y='percentage', markers=True,
                       title='Impact of Education on Food Insecurity',
                       labels={'education_level': 'Education Level', 'percentage': 'Food Insecurity Rate (%)'},
                       height=400)
        fig3.update_traces(line=dict(color='#3B82F6', width=3), 
                           marker=dict(size=12, color='#EF4444', line=dict(width=2, color='#FFFFFF')))
        fig3.update_layout(margin=dict(t=50, b=0, l=0, r=0), plot_bgcolor='rgba(0,0,0,0)', 
                           yaxis=dict(gridcolor='#E5E7EB'), xaxis=dict(showgrid=False))
        st.plotly_chart(fig3, use_container_width=True)
        
    st.markdown("#### Urban vs Rural Asset Gap Segmented by Security")
    fig6 = px.box(df.dropna(subset=['asset_score', 'location_type']), x='location_type', y='asset_score', color='is_food_insecure',
                  labels={'location_type': 'Location Profile', 'asset_score': 'Asset Ownership Score'},
                  color_discrete_map={0: '#10B981', 1: '#EF4444'}, height=350)
    fig6.for_each_trace(lambda t: t.update(name = 'Secure' if t.name == '0' else 'Insecure'))
    fig6.update_layout(margin=dict(t=20, b=0, l=0, r=0), plot_bgcolor='rgba(0,0,0,0)', 
                       yaxis=dict(gridcolor='#E5E7EB'))
    st.plotly_chart(fig6, use_container_width=True)


with tab3:
    st.markdown("### Multidimensional Deprivations")
    colE, colF = st.columns(2)
    
    # Pre-calculations for deprivations
    for code, name in DEPRIVATION_VARS.items():
        if code in df.columns:
            df[f'{name}_binary'] = (df[code] > 0).astype(int)
    
    deprivation_cols = ['Q6A', 'Q6B', 'Q6C', 'Q6E']
    if all(col in df.columns for col in deprivation_cols):
        dep_data = df[deprivation_cols].copy()
        dep_data.columns = ['Food', 'Water', 'Medical Care', 'Cash Income']
        corr_matrix = dep_data.corr().round(2)
        
        with colE:
            st.markdown("<p style='text-align:center; font-weight:600; color:#374151'>Deprivation Correlation Matrix</p>", unsafe_allow_html=True)
            # Create a heatmap without the upper triangle if desired, or keep simple standard
            fig4 = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Reds',
                             aspect='auto', height=400)
            fig4.update_layout(margin=dict(t=20, b=0, l=0, r=0))
            st.plotly_chart(fig4, use_container_width=True)

        with colF:
            st.markdown("<p style='text-align:center; font-weight:600; color:#374151'>Vulnerability Spread (Count of Deprivations)</p>", unsafe_allow_html=True)
            df['deprivation_count'] = (df[deprivation_cols] > 0).sum(axis=1)
            dep_dist = df['deprivation_count'].value_counts().reset_index()
            dep_dist.columns = ['Count', 'Households']
            dep_dist['Count'] = dep_dist['Count'].astype(int).astype(str) + " Deprivations"
            
            fig5 = px.bar(dep_dist, x='Households', y='Count', orientation='h',
                          color='Count', color_discrete_sequence=px.colors.sequential.Reds, height=400)
            
            fig5.update_layout(margin=dict(t=20, b=0, l=0, r=0), showlegend=False, 
                               plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(gridcolor='#E5E7EB'), 
                               yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig5, use_container_width=True)
