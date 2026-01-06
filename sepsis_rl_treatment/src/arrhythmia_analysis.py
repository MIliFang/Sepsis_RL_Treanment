import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
import seaborn as sns
from collections import defaultdict

from config import Config


def plot_usage_by_arrhythmia(samples, model, decoder, fig_path, logger):
    """
    Compare dopamine usage between clinicians and AI model, stratified by heart rate (tachycardia vs normal).
    """
    
    # 1. Split samples by heart rate (tachycardia: HR >= 100)
    tachy_samples = []   # HR >= 100
    normal_samples = []  # HR < 100

    for s in samples:
        hr = s['current_clinical'].get('heart_rate', 0)
        if hr >= 100:
            tachy_samples.append(s)
        else:
            normal_samples.append(s)
            
    logger.info(f"Tachycardia Analysis: HR>=100 (n={len(tachy_samples)}), HR<100 (n={len(normal_samples)})")

    # 2. Extract dopamine usage for a given sample group
    def get_dopa_usage(sample_list, label_name):
        """Return DataFrame with dopamine usage (0/1) for both clinician and AI."""
        if not sample_list:
            # Return empty structure to avoid concat/pivot errors
            return pd.DataFrame(columns=['Group', 'Source', 'Dopamine_Used'])
        
        phy_dopa = []
        ai_dopa = []
        
        model.eval()
        for s in sample_list:
            # Clinician action
            doses_phy = decoder.decode(s['action'])
            phy_dopa.append(1 if doses_phy.get('dopa_max', 0) > 0 else 0)
            
            # AI action
            state = torch.FloatTensor(s['state']).unsqueeze(0).to(Config.DEVICE)
            with torch.no_grad():
                aid, _ = model.select_action(state, epsilon=0, deterministic=True)
            doses_ai = decoder.decode(aid)
            ai_dopa.append(1 if doses_ai.get('dopa_max', 0) > 0 else 0)
        
        # Create DataFrames (clinician first for consistent plotting order)
        df_ai = pd.DataFrame({'Group': label_name, 'Source': 'AI Model', 'Dopamine_Used': ai_dopa})
        df_phy = pd.DataFrame({'Group': label_name, 'Source': 'Clinician', 'Dopamine_Used': phy_dopa})
        return pd.concat([df_phy, df_ai])

    # 3. Build dataset for both groups
    label_tachy = f"HR≥100\n(n={len(tachy_samples)})"
    label_norm = f"HR<100\n(n={len(normal_samples)})"

    df1 = get_dopa_usage(tachy_samples, label_tachy)
    df2 = get_dopa_usage(normal_samples, label_norm)

    # Handle empty groups by providing zero-filled placeholders
    if df1.empty:
        df1 = pd.DataFrame({
            'Group': [label_tachy, label_tachy], 
            'Source': ['Clinician', 'AI Model'],  
            'Dopamine_Used': [0.0, 0.0]
        })
    if df2.empty:
        df2 = pd.DataFrame({
            'Group': [label_norm, label_norm], 
            'Source': ['Clinician', 'AI Model'],  
            'Dopamine_Used': [0.0, 0.0]
        })

    full_df = pd.concat([df1, df2], ignore_index=True)
    
    # Compute average usage rate per group and source
    plot_data = full_df.groupby(['Group', 'Source'])['Dopamine_Used'].mean().reset_index()

    # 4. Plot results
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    # Define consistent colors
    color_map = {
        'Clinician': 'steelblue',
        'AI Model': 'darkorange'
    }
    
    # Enforce plotting order: Clinician first, then AI
    plot_data['Source'] = pd.Categorical(plot_data['Source'], categories=['Clinician', 'AI Model'], ordered=True)
    plot_data = plot_data.sort_values('Source')
    
    # Create grouped bar plot
    ax = sns.barplot(
        x='Group', 
        y='Dopamine_Used', 
        hue='Source', 
        data=plot_data, 
        palette=color_map,
        hue_order=['Clinician', 'AI Model']
    )
    
    plt.title('Dopamine Usage Rate by Heart Rate', fontsize=12, pad=15)
    plt.ylabel('Dopamine Usage Rate', fontsize=10)
    plt.xlabel('')
    
    # Adjust y-axis to leave space for labels
    max_val = plot_data['Dopamine_Used'].max()
    if max_val == 0: 
        max_val = 0.001  # Avoid flat plot when all values are zero
    plt.ylim(0, max_val * 1.25)

    # Add percentage labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', padding=3, fontsize=10)

    # Position legend
    plt.legend(loc='upper right', title=None)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(fig_path, 'arrhythmia_dopa_usage.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Dopamine usage by arrhythmia status saved to {save_path}")
    return plot_data.to_dict()