import os
import json
import numpy as np
import pandas as pd
import random
import torch
import torch.optim as optim
from collections import defaultdict
import logging
import time  
from datetime import datetime

# Import custom modules
from config import Config
from data_processor import extract_high_frequency_actions, DataProcessor
from physician_policy import PhysicianPolicy
from rl_models import MTLDuelingQLearning, MTLAttentionCQL, MTL_RLTrainer
from OPE import OPEE
from setup import setup_logger, setup_seed
from arrhythmia_analysis import plot_usage_by_arrhythmia


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def convert_to_serializable(obj):
    """
    Recursively convert NumPy and PyTorch types to native Python types
    to ensure JSON serializability.
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, torch.Tensor):
        return convert_to_serializable(obj.detach().cpu().numpy())
    else:
        return obj


def save_experiment_results(results, save_dir):
    """Save full experiment results as a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f"experiment_results_{timestamp}.json")
    serializable_results = convert_to_serializable(results)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=4)
    return save_path


def save_training_history(history, model_name, save_dir):
    """Save training history (loss, reward, etc.) as JSON."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f"{model_name}_training_history_{timestamp}.json")
    serializable_history = convert_to_serializable(history)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_history, f, ensure_ascii=False, indent=4)
    return save_path


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    # 1. Basic setup
    Config.create_output_dirs()
    log_dir = os.path.join(Config.LOG_PATH, 'rl_trainer')
    logger = setup_logger(log_dir)
    setup_seed(Config.RANDOM_SEED)
    
    logger.info("="*60)
    logger.info(">>> Starting Reinforcement Learning Training (Sepsis ICU) <<<")
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Data Path: {Config.DATA_PATH}")
    logger.info("="*60)
    
    experiment_start_time = time.time()

    # 2. Load and preprocess data
    logger.info("Loading MIMIC-III data...")
    df = pd.read_csv(Config.DATA_PATH)

    # Step 1: Extract high-frequency clinically plausible actions
    high_freq_combos = extract_high_frequency_actions(df, min_freq=2, logger=logger)

    # Step 2: Build dataset with valid action space and state features
    data_processor = DataProcessor(logger=logger, high_freq_combos=high_freq_combos)
    data_processor.preprocess(df) 
    data_split = data_processor.split_data()

    # Get dimensions
    state_dim = data_split['state_dim']
    valid_actions = data_split['valid_actions']
    action_dim = len(valid_actions)
    
    logger.info(f"State Dim: {state_dim} | Action Dim: {action_dim}")
    logger.info(f"Train: {len(data_split['train_trans'])} steps | Val: {len(data_split['val'])} steps")

    # Build physician policy for OPE and comparison
    physician_policy_agent = PhysicianPolicy(
        physician_policy=data_split['physician_policy'], 
        valid_actions=valid_actions, 
        logger=logger
    )
    
    # Initialize results container
    experiment_results = {
        'meta': {
            'seed': Config.RANDOM_SEED,
            'epochs': Config.TRAIN_EPOCHS,
            'lr': Config.LEARNING_RATE
        },
        'models': {},
        'ope': {},
        'interpretation': {}
    }

    # -----------------------------------------------------
    # 3. Train Model 1: MTL Dueling Q-Learning
    # -----------------------------------------------------
    logger.info("\n>>> [Model 1] Training MTL Dueling Q-Learning...")
    mtl_dueling = MTLDuelingQLearning(state_dim, action_dim).to(Config.DEVICE)
    mtl_dueling.set_data_processor(data_processor)
    
    opt_dueling = optim.AdamW(mtl_dueling.parameters(), lr=Config.LEARNING_RATE)
    trainer_dueling = MTL_RLTrainer(mtl_dueling, opt_dueling, logger, data_processor, enable_cql=False)
    
    t0 = time.time()
    hist_dueling = trainer_dueling.train(data_split['train_trans'], data_split['val'], epochs=Config.TRAIN_EPOCHS)
    dur_dueling = time.time() - t0
    
    # Save model and history
    save_training_history(hist_dueling, 'MTL_DuelingQL', Config.OUTPUT_DATA_PATH)
    torch.save(mtl_dueling.state_dict(), os.path.join(Config.OUTPUT_MODEL_PATH, "mtl_dueling_best.pth"))
    
    experiment_results['models']['MTL_DuelingQL'] = {
        'duration': dur_dueling,
        'best_val_reward': trainer_dueling.best_val_reward
    }

    # -----------------------------------------------------
    # 4. Train Model 2: MTL Attention CQL (main model)
    # -----------------------------------------------------
    logger.info("\n>>> [Model 2] Training MTL Attention CQL...")
    mtl_cql = MTLAttentionCQL(state_dim, action_dim).to(Config.DEVICE)
    mtl_cql.set_data_processor(data_processor)
    
    opt_cql = optim.AdamW(mtl_cql.parameters(), lr=Config.LEARNING_RATE)
    trainer_cql = MTL_RLTrainer(mtl_cql, opt_cql, logger, data_processor, enable_cql=True)
    
    t0 = time.time()
    hist_cql = trainer_cql.train(data_split['train_trans'], data_split['val'], epochs=Config.TRAIN_EPOCHS)
    dur_cql = time.time() - t0
    
    # Save model and history
    save_training_history(hist_cql, 'MTL_AttentionCQL', Config.OUTPUT_DATA_PATH)
    torch.save(mtl_cql.state_dict(), os.path.join(Config.OUTPUT_MODEL_PATH, "mtl_cql_best.pth"))
    
    experiment_results['models']['MTL_AttentionCQL'] = {
        'duration': dur_cql,
        'best_val_reward': trainer_cql.best_val_reward
    }

    # -----------------------------------------------------
    # 5. Off-Policy Evaluation (OPE)
    # -----------------------------------------------------
    logger.info("\n>>> Off-Policy Evaluation (OPE)...")
    
    def group_into_trajectories(flat_samples):
        """Group flat transitions into per-patient trajectories."""
        pid_map = defaultdict(list)
        for s in flat_samples:
            pid = s.get('icustayid')
            pid_map[pid].append(s)
        trajs = []
        for pid in pid_map:
            trajs.append(sorted(pid_map[pid], key=lambda x: x['timestep']))
        return trajs

    try:
        ope_eval = OPEE(Config.OUTPUT_FIG_PATH)
        
        # Compute baseline (physician) return
        val_trajs = group_into_trajectories(data_split['val'])
        baseline_returns = []
        for traj in val_trajs:
            ret = sum((Config.GAMMA**i) * step['reward'] for i, step in enumerate(traj))
            baseline_returns.append(ret)
        baseline_avg = np.mean(baseline_returns) if baseline_returns else 0.0

        # Evaluate models using WIS
        res_dueling_full = ope_eval.evaluate(mtl_dueling, physician_policy_agent, data_split['val'])
        res_cql_full = ope_eval.evaluate(mtl_cql, physician_policy_agent, data_split['val'])

        print(f"{'Physician (Baseline)':<25}  {baseline_avg:<12.4f}")
        
        wis_d = res_dueling_full.get('wis_score', 0.0)
        ess_d = res_dueling_full.get('ess', 0.0)
        lift_d = wis_d / baseline_avg if baseline_avg != 0 else 0
        
        wis_c = res_cql_full.get('wis_score', 0.0)
        ess_c = res_cql_full.get('ess', 0.0)
        lift_c = wis_c / baseline_avg if baseline_avg != 0 else 0

        # Plot and save OPE analysis
        results_map = {
            'MTL_DuelingQL': res_dueling_full,
            'MTL_AttentionCQL': res_cql_full
        }
        ope_eval.plot_wis_analysis(results_map, baseline_reward=baseline_avg)
        
        # Save OPE results
        experiment_results['ope'] = {
            'baseline': baseline_avg,
            'MTL_DuelingQL': res_dueling_full,
            'MTL_AttentionCQL': res_cql_full
        }
        
    except Exception as e:
        logger.error(f"OPE evaluation failed: {e}", exc_info=True)

    # -----------------------------------------------------
    # 6. Tachycardia-Specific Analysis (HR ≥ 100)
    # -----------------------------------------------------
    has_heart_rate = any('current_clinical' in s and 'heart_rate' in s['current_clinical'] for s in data_split['test'])
    if has_heart_rate:
        logger.info(">>> Analyzing dopamine usage in tachycardic patients (HR ≥ 100)...")
        try:
            plot_data = plot_usage_by_arrhythmia(
                data_split['test'], mtl_cql, data_processor.action_decoder, 
                Config.OUTPUT_FIG_PATH, logger
            )
            experiment_results['tachycardia_analysis'] = {
                'dopamine_usage_plot': plot_data,
            }
            logger.info("Tachycardia analysis completed")
        except Exception as e:
            logger.error(f"Tachycardia analysis failed: {e}", exc_info=True)

    # -----------------------------------------------------
    # 7. Finalize and Save Experiment Summary
    # -----------------------------------------------------
    try:
        experiment_end_time = time.time()
        total_duration = experiment_end_time - experiment_start_time
        
        experiment_results['experiment_info'] = {
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_duration_seconds': total_duration,
            'total_duration_minutes': total_duration / 60
        }
        
        results_path = save_experiment_results(experiment_results, Config.OUTPUT_DATA_PATH)
        logger.info(f"Full experiment results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Failed to save final results: {str(e)}", exc_info=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Uncaught exception: {e}", exc_info=True)