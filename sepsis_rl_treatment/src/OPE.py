import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from config import Config


class OPEE:
    """Off-Policy Evaluator (OPE) using Clipped Weighted Importance Sampling (WIS)."""
    
    def __init__(self, save_path):
        self.save_path = save_path
        self.device = Config.DEVICE
        os.makedirs(save_path, exist_ok=True)
    
    def evaluate(self, agent, behavior_policy, samples, gamma=Config.GAMMA, epsilon=0.01):
        """
        Evaluate a target policy using Clipped Weighted Importance Sampling (WIS).
        """
        agent.eval()
        
        # Get WIS clipping bounds from config
        clip_min = getattr(Config, 'WIS_CLIP_MIN', 0.0)
        clip_max = getattr(Config, 'WIS_CLIP_MAX', 20.0) 
        
        # Group samples into patient trajectories
        trajectories = defaultdict(list)
        for s in samples:
            uid = s.get('icustayid')
            trajectories[uid].append(s)
        
        for uid in trajectories:
            trajectories[uid].sort(key=lambda x: x.get('step', 0))
            
        wis_returns = []   # Weighted returns per trajectory
        weights = []       # Cumulative importance weights
        rhos_all = []      # All per-step importance ratios (for diagnostics)
        
        with torch.no_grad():
            for uid, traj in trajectories.items():
                rho_trajectory = 1.0   # Cumulative IS weight for this trajectory
                G_trajectory = 0.0     # Discounted return
                valid_trajectory = True
                
                for t, step in enumerate(traj):
                    # Prepare state tensor
                    state_np = step['state']
                    state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)
                    action_idx = step.get('action_idx', step.get('action'))
                    reward = step.get('reward', 0.0)
                    
                    # A. Compute target policy probability π_e(a|s)
                    q_values, _ = agent(state_tensor)
                    pi_probs = F.softmax(q_values / 1.0, dim=1).cpu().numpy()[0]
                    pi_prob = pi_probs[action_idx] if action_idx < len(pi_probs) else epsilon
                    
                    # B. Compute behavior policy probability π_b(a|s)
                    if hasattr(behavior_policy, 'get_action_prob'):
                        b_prob = behavior_policy.get_action_prob(state_np, action_idx)
                    else:
                        behavior_dist = behavior_policy._get_state_distribution(state_np)
                        b_prob = behavior_dist.get(action_idx, epsilon)
                    
                    b_prob = max(b_prob, 1e-3)  # Avoid division by zero
                    
                    # C. Compute per-step importance sampling ratio
                    rho = pi_prob / b_prob
                    rho_clipped = min(max(rho, clip_min), clip_max)
                    
                    rhos_all.append(rho_clipped)
                    rho_trajectory *= rho_clipped
                    G_trajectory += (gamma ** t) * reward
                    
                    # Early termination if weight becomes unstable
                    if rho_trajectory > 1e4:
                        rho_trajectory = 1e4
                    if rho_trajectory < 1e-8:
                        valid_trajectory = False
                        break
                
                # Accumulate WIS components
                if valid_trajectory:
                    weights.append(rho_trajectory)
                    wis_returns.append(rho_trajectory * G_trajectory)
                else:
                    weights.append(0.0)
                    wis_returns.append(0.0)

        # Compute final WIS score
        sum_weights = np.sum(weights)
        if sum_weights < 1e-6:
            wis_score = 0.0
        else:
            wis_score = np.sum(wis_returns) / sum_weights
            
        sum_sq_weights = np.sum(np.square(weights))
        ess = (sum_weights ** 2) / (sum_sq_weights + 1e-8) if sum_sq_weights > 0 else 0
        
        # Coverage: fraction of trajectories with non-zero weight
        non_zero_weights = np.sum(np.array(weights) > 1e-6)
        coverage = non_zero_weights / len(trajectories) if len(trajectories) > 0 else 0
        
        results = {
            'wis_score': wis_score,
            'ess': ess,  # Keep ESS for logging or analysis
            'coverage': coverage,
            'rho_distribution': rhos_all,
            'n_trajectories': len(trajectories)
        }
        
        return results

    def plot_wis_analysis(self, wis_results, baseline_reward=None):
        """
        Plot WIS evaluation results: (a) WIS scores, (b) trajectory coverage, (c) importance weight distribution.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 3 columns
        
        models = list(wis_results.keys())
        wis_scores = [wis_results[m]['wis_score'] for m in models]
        
        # (a) WIS Scores
        colors = ['#3498db' if i != 0 else '#e74c3c' for i in range(len(models))]
        bars1 = axes[0].bar(models, wis_scores, color=colors, alpha=0.8)
        axes[0].set_title('(a) WIS Scores (Est. Return)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('WIS Score')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x')
        axes[0].set_xticklabels(models, ha='right')
        
        if baseline_reward is not None:
            axes[0].axhline(y=baseline_reward, color='#2c3e50', linestyle='--', 
                            linewidth=2, alpha=0.7, label=f'Baseline: {baseline_reward:.2f}')
            axes[0].legend()
            
        for bar, score in zip(bars1, wis_scores):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{score:.2f}', ha='center', va='bottom', fontsize=10)

        # (b) Trajectory Coverage
        cov_values = [wis_results[m]['coverage'] for m in models]
        bars3 = axes[1].bar(models, cov_values, color='#9b59b6', alpha=0.8)
        axes[1].set_title('(b) Trajectory Coverage', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Coverage')
        axes[1].set_ylim(0, 1.1)
        axes[1].tick_params(axis='x')
        axes[1].set_xticklabels(models, ha='right')
        
        for bar, val in zip(bars3, cov_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10)

        # (c) Importance Weight Distribution
        for i, model in enumerate(models):
            if 'rho_distribution' in wis_results[model]:
                rhos = wis_results[model]['rho_distribution']
                rhos_filtered = [r for r in rhos if r < 5.0]  # Exclude extreme weights
                axes[2].hist(rhos_filtered, bins=30, alpha=0.5, label=model, density=True)
        
        axes[2].set_title('(c) Importance Weight Dist (Clipped < 5)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Importance Ratio (ρ)')
        axes[2].set_ylabel('Density')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Final layout and save
        plt.suptitle('Off-Policy Evaluation Analysis', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_path, 'wis_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path