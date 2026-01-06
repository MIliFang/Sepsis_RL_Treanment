import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import random
import types 
from collections import Counter
from scipy.spatial.distance import jensenshannon

from config import Config
from data_processor import DataProcessor, extract_high_frequency_actions
from rl_models import MTLDuelingQLearning, MTLAttentionCQL, MTL_RLTrainer
from setup import setup_logger, setup_seed

VAL_SPLIT = 0.2
CQL_ALPHA_VAL = 2.0  # Renamed to avoid conflict

def inject_simple_cql_loss(model):
    """
    Dynamically inject a CQL loss method into an MLP model instance.
    Enables CQL regularization for models that don't natively support it.
    """
    def compute_cql_loss(self, states, actions, q_values):
        # CQL Regularization: minimize logsumexp(Q) - Q(s,a)
        logsumexp = torch.logsumexp(q_values, dim=1).mean()
        q_at_actions = q_values.gather(1, actions.unsqueeze(1)).mean()
        return Config.CQL_ALPHA * (logsumexp - q_at_actions)
    
    model.compute_cql_loss = types.MethodType(compute_cql_loss, model)
    logger.info(">>> Injected CQL loss function into MLP model")


def evaluate_clinical_consistency(model, val_samples, data_processor):
    """
    Evaluate how closely the AI policy matches clinician behavior.
    Returns:
        - accuracy: action matching rate
        - avg_q: average predicted Q-value
        - jsd: Jensen-Shannon divergence between action distributions
    """
    model.eval()
    matches = 0
    total = 0
    q_values_sum = 0
    
    doc_actions = []
    ai_actions = []
    
    valid_action_indices = list(range(len(data_processor.valid_actions)))

    with torch.no_grad():
        for sample in val_samples:
            state = sample['state']
            doc_action_idx = sample.get('action_idx', 0)
            
            ai_action_idx, _ = model.select_action(
                state, epsilon=0.0, training=False, deterministic=True
            )
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            q_values, _ = model.forward(state_tensor)
            q_values_np = q_values.cpu().numpy().flatten()
            max_q = q_values_np[valid_action_indices].max()
            
            if ai_action_idx == doc_action_idx:
                matches += 1
            
            doc_actions.append(doc_action_idx)
            ai_actions.append(ai_action_idx)
            q_values_sum += max_q
            total += 1
            
    # 1. Behavioral Cloning Accuracy
    accuracy = matches / total if total > 0 else 0
    
    # 2. Average Q-Value (safety indicator)
    avg_q = q_values_sum / total if total > 0 else 0

    # 3. Jensen-Shannon Divergence (distribution similarity)
    n_actions = len(data_processor.valid_actions)
    doc_counts = Counter(doc_actions)
    ai_counts = Counter(ai_actions)
    
    doc_dist = np.array([doc_counts.get(i, 0) for i in range(n_actions)])
    ai_dist = np.array([ai_counts.get(i, 0) for i in range(n_actions)])
    
    doc_prob = doc_dist / (doc_dist.sum() + 1e-10)
    ai_prob = ai_dist / (ai_dist.sum() + 1e-10)
    
    js_distance = jensenshannon(doc_prob, ai_prob)
    
    logger.info(f"Action Distribution Divergence (JSD): {js_distance:.4f}")
    
    return accuracy, avg_q, js_distance
    

def run_experiment(exp_name, config, data_split, data_processor, epochs=20):
    """Run a single ablation experiment."""
    logger.info(f"\n{'='*20} Running {exp_name} {'='*20}")
    
    setup_seed(Config.RANDOM_SEED)
    state_dim = data_split['state_dim']
    action_dim = len(data_split['valid_actions'])
    
    # Configure CQL based on experiment settings
    if config['use_cql']:
        Config.CQL_ALPHA = CQL_ALPHA_VAL  # Strong constraint
        logger.info(f"CQL Enabled: Alpha set to {Config.CQL_ALPHA}")
    else:
        Config.CQL_ALPHA = 0.0
        logger.info("CQL Disabled: Alpha set to 0.0")

    # Initialize model
    if config['model_type'] == 'MLP':
        model = MTLDuelingQLearning(state_dim, action_dim).to(Config.DEVICE)
        if config['use_cql'] and not hasattr(model, 'compute_cql_loss'):
            inject_simple_cql_loss(model)
    elif config['model_type'] == 'Attention':
        model = MTLAttentionCQL(state_dim, action_dim).to(Config.DEVICE)
    else:
        raise ValueError("Unknown model type")
        
    model.set_data_processor(data_processor)
    
    # Initialize trainer
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    trainer = MTL_RLTrainer(
        model, optimizer, logger, data_processor, 
        device=Config.DEVICE,
        enable_mtl=config['use_mtl'], 
        enable_cql=config['use_cql']
    )
    
    # Train model
    history = trainer.train(
        data_split['train_trans'], 
        data_split['val'], 
        epochs=epochs,
        save_best=False 
    )
    
    # Evaluate consistency
    acc, avg_q, jsd = evaluate_clinical_consistency(model, data_split['val'], data_processor)
    
    logger.info(f"[{exp_name}] Final Clinical Accuracy: {acc:.2%}")
    logger.info(f"[{exp_name}] Average Predicted Q-Value: {avg_q:.2f}")
    logger.info(f"[{exp_name}] Distribution JSD: {jsd:.4f}")
    
    history['final_accuracy'] = acc
    history['final_avg_q'] = avg_q
    history['final_jsd'] = jsd
    
    return history


# Unified color palette (blue only)
COLORS = {
    'primary': '#1f77b4',    # Standard matplotlib blue
    'light': '#aec7e8',      # Light blue
    'dark': '#0055a4'        # Dark blue
}

def plot_ablation_results(results, save_path):
    """Plot all metrics using bar charts with consistent blue color scheme."""
    
    os.makedirs(save_path, exist_ok=True) 
    
    # Set up academic-style plotting
    sns.set(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Prepare data
    exp_names = list(results.keys())
    plot_data = []
    
    for exp_name in exp_names:
        # Format experiment labels
        if 'Exp1' in exp_name:
            exp_label = 'Exp1 (MLP)'
        elif 'Exp2' in exp_name:
            exp_label = 'Exp2 (+MTL)'
        elif 'Exp3' in exp_name:
            exp_label = 'Exp3 (+CQL)'
        elif 'Exp4' in exp_name:
            exp_label = 'Exp4 (Attn)'
        else:
            exp_label = exp_name
        
        jsd = results[exp_name]['final_jsd']
        avg_q = results[exp_name]['final_avg_q']
        acc = results[exp_name]['final_accuracy'] * 100  # Convert to percentage
        
        plot_data.append({
            'Experiment': exp_label,
            'Accuracy (%)': acc,
            'JSD': jsd,
            'Avg Q-Value': avg_q
        })
    
    df = pd.DataFrame(plot_data)
    
    # Create bar plots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel A: Clinical Accuracy (Higher is better)
    ax1 = axes[0]
    bars1 = ax1.bar(df['Experiment'], df['Accuracy (%)'], color=COLORS['primary'], alpha=0.85)
    ax1.set_title('(A) Clinical Policy Accuracy\n(Higher is Better)', fontweight='bold', pad=15)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Panel B: JSD Divergence (Lower is better)
    ax2 = axes[1]
    bars2 = ax2.bar(df['Experiment'], df['JSD'], color=COLORS['primary'], alpha=0.85)
    ax2.set_title('(B) Distributional Divergence\n(Lower is Better)', fontweight='bold', pad=15)
    ax2.set_ylabel('Jensen-Shannon Divergence')
    ax2.set_ylim(0, max(df['JSD']) * 1.2)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Panel C: Q-Value Estimation (Safety check)
    ax3 = axes[2]
    bars3 = ax3.bar(df['Experiment'], df['Avg Q-Value'], color=COLORS['primary'], alpha=0.85)
    ax3.set_title('(C) Q-Value Estimation\n(Conservative is Safer)', fontweight='bold', pad=15)
    ax3.set_ylabel('Average Q-Value')
    ax3.set_ylim(0, max(df['Avg Q-Value']) * 1.2)
    
    # Add safety threshold line
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Safety Threshold')
    ax3.legend()
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Rotate x-axis labels for readability
    for ax in axes:
        ax.tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(save_path, "Ablation_Comparative_Analysis.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Bar chart visualization saved to: {fig_path}")

    # Save numerical results
    exp_names_clean = [name.split(':')[0] if ':' in name else name for name in results.keys()]
    accuracies = [results[k]['final_accuracy']*100 for k in results.keys()]
    avg_qs = [results[k]['final_avg_q'] for k in results.keys()]
    jsds = [results[k]['final_jsd'] for k in results.keys()]
    final_rewards = [np.mean(results[k]['val_reward'][-5:]) for k in results.keys()]
    
    results_df = pd.DataFrame({
        'Experiment': exp_names_clean,
        'Final_Accuracy(%)': accuracies,
        'Average_Q_Value': avg_qs,
        'JSD_Divergence': jsds,
        'Final_Avg_Reward': final_rewards
    })
    csv_path = os.path.join(save_path, 'ablation_results.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Numerical results saved to: {csv_path}")


def main():
    """Main function to run ablation study."""
    Config.create_output_dirs()

    global logger
    log_dir = os.path.join(Config.LOG_PATH, 'ablation')  
    logger = setup_logger(log_dir)
    
    logger.info("Loading Data...")
    try:
        df = pd.read_csv(Config.DATA_PATH)
    except FileNotFoundError:
        logger.error(f"Data file not found at {Config.DATA_PATH}")
        return
    
    # Extract high-frequency actions
    high_freq_combos = extract_high_frequency_actions(df, min_freq=2, logger=logger)

    # Preprocess data
    data_processor = DataProcessor(logger=logger, high_freq_combos=high_freq_combos)
    samples = data_processor.preprocess(df)
    data_split = data_processor.split_data(val_split=VAL_SPLIT)
    
    # Define ablation experiments
    experiments = {
        'Exp1: Baseline (MLP)': {
            'model_type': 'MLP',
            'use_mtl': False,
            'use_cql': False
        },
        'Exp2: + MTL': {
            'model_type': 'MLP',
            'use_mtl': True,
            'use_cql': False
        },
        'Exp3: + CQL': {
            'model_type': 'MLP',
            'use_mtl': True,
            'use_cql': True
        },
        'Exp4: + Attention (Full)': {
            'model_type': 'Attention',
            'use_mtl': True,
            'use_cql': True
        }
    }
    
    all_results = {}
    EPOCHS = 20
    
    for exp_name, config in experiments.items():
        try:
            history = run_experiment(exp_name, config, data_split, data_processor, epochs=EPOCHS)
            all_results[exp_name] = history
        except Exception as e:
            logger.error(f"Failed to run {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            
    if all_results:
        logger.info("Plotting results...")
        plot_ablation_results(all_results, Config.OUTPUT_FIG_PATH)
        logger.info(f"Done! Results saved to {Config.OUTPUT_FIG_PATH}")


if __name__ == "__main__":
    main()
