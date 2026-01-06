import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler

from config import Config


def discretize(
        val: float, 
        drug_name: str,
        clinical_ranges: Dict[str, Tuple[float, ...]] = Config.CLINICAL_RANGES
        ) -> int:
    """Discretize continuous drug dosage into ordinal levels (0 = none, 1–3 = low–high)."""
    if pd.isna(val) or val <= 0:
        return 0
    thresholds = clinical_ranges.get(drug_name)
    if thresholds is None:
        return 0

    if drug_name == 'dopa_max':
        return 1 if val > thresholds[0] else 0
    elif len(thresholds) == 3:  # e.g., AVP: (0, 0.05, 0.1) → L0, L1, L2
        if val <= thresholds[1]:
            return 1
        elif val <= thresholds[2]:
            return 2
        else:
            return 2  # clamp to highest valid level
    else:  # 4 thresholds → L1, L2, L3
        if val <= thresholds[1]:
            return 1
        elif val <= thresholds[2]:
            return 2
        else:
            return 3


def extract_high_frequency_actions(
    df: pd.DataFrame, 
    min_freq: int = 5, 
    logger=None,
) -> List[Tuple[int, int, int, int, int]]:
    """
    Extract clinically plausible drug dosage combinations (discretized) that appear frequently in data.
    """
    drug_cols = ['norad_max', 'vaso_max', 'epi_max', 'dopa_max', 'phenyl_max']
    for col in drug_cols:
        if col not in df.columns:
            df[col] = 0.0

    discretized = []
    for _, row in df[drug_cols].iterrows():
        ne = discretize(row['norad_max'], 'norad_max')
        avp = discretize(row['vaso_max'], 'vaso_max')
        epi = discretize(row['epi_max'], 'epi_max')
        dopa = discretize(row['dopa_max'], 'dopa_max')
        phenyl = discretize(row['phenyl_max'], 'phenyl_max')
        
        # Enforce clinical constraints consistent with ActionDecoder
        if ne == 1:
            avp = min(avp, 2)
            epi = min(epi, 2)
            phenyl = min(phenyl, 2)
            dopa = 0
        elif ne == 2:
            phenyl = min(phenyl, 2)
            dopa = min(dopa, 1)
        elif ne >= 3:
            epi = 0
            dopa = min(dopa, 2)  
        
        discretized.append((ne, avp, epi, dopa, phenyl))
    
    counter = Counter(discretized)
    high_freq = [combo for combo, cnt in counter.items() if cnt >= min_freq]
    high_freq = sorted(high_freq, key=lambda x: counter[x], reverse=True)
    
    print(f"[Action Discovery] Found {len(high_freq)} clinically plausible action combos (min_freq={min_freq})")
    for i, combo in enumerate(high_freq[:10]):
        print(f"  Top {i+1}: {combo} → {counter[combo]} times")
    
    return high_freq


def apply_clinical_constraints(
    ne: int, 
    avp: int, 
    epi: int, 
    dopa: int, 
    phe: int
) -> Tuple[int, int, int, int, int]:
    """
    Apply clinical dosing constraints to ensure realistic drug combinations.
    
    Rules:
    - NE=L1: AVP≤2, EPI≤2, PHE≤2, DOPA=0
    - NE=L2: PHE≤2, DOPA≤1
    - NE≥L3: EPI=0, DOPA≤2
    """
    ne = max(0, int(ne))
    avp = max(0, int(avp))
    epi = max(0, int(epi))
    dopa = max(0, int(dopa))
    phe = max(0, int(phe))
    
    if ne == 1:
        avp = min(avp, 2)
        epi = min(epi, 2)      
        phe = min(phe, 2)      
        dopa = 0
    elif ne == 2:
        phe = min(phe, 2)
        dopa = min(dopa, 1)
    elif ne >= 3:
        epi = 0                
        dopa = min(dopa, 2)    
    
    return (ne, avp, epi, dopa, phe)


class ActionDecoder:
    """
    Maps discrete action IDs to clinically valid drug dosage combinations.
    Combines rule-based and data-driven action spaces.
    """
    def __init__(
        self, 
        logger: None,
        drug_mapping: Dict[str, str] = Config.DRUG_MAPPING,
        high_freq_combos: Optional[List[Tuple[int, ...]]] = None,
        rule_based_combos: Optional[List[Tuple[int, ...]]] = None
    ):
        self.logger = logger
        self.drug_mapping = drug_mapping
        self.drug_names = list(drug_mapping.keys())
        
        # Generate rule-based action space
        if rule_based_combos is None:
            rule_based_combos = self._get_rule_based_combos()
        rule_set = set(rule_based_combos)
        
        # Use data-driven combinations if provided
        data_set = set(high_freq_combos) if high_freq_combos else set()
        
        # Final action space = intersection of rule-based and frequent data combos
        if not data_set:
            if self.logger:
                self.logger.info("⚠️ Warning: No data-driven actions provided. Using full rule-based set.")
            all_combos = rule_set
        else:
            intersection = data_set.intersection(rule_set)
            if self.logger:
                self.logger.info(f"[Action Space Construction]")
                self.logger.info(f"  - Data-driven unique actions: {len(data_set)}")
                self.logger.info(f"  - Rule-based unique actions:  {len(rule_set)}")
                self.logger.info(f"  - Intersection (Final Space): {len(intersection)}")
            
            if len(intersection) == 0:
                if self.logger:
                    self.logger.info("⚠️ Critical Warning: Intersection is empty! Fallback to Data set.")
                all_combos = data_set
            else:
                all_combos = intersection
        
        # Build mapping tables
        all_combos = sorted(list(all_combos))
        self.action_table: Dict[int, Dict[str, int]] = {}
        self.dose_tuple_to_action: Dict[Tuple[int, ...], int] = {}
        
        for action_id, (ne, avp, epi, dopa, phenyl) in enumerate(all_combos):
            dose_dict = {
                'norad_max': ne,
                'vaso_max': avp,
                'epi_max': epi,
                'dopa_max': dopa,
                'phenyl_max': phenyl
            }
            self.action_table[action_id] = dose_dict
            self.dose_tuple_to_action[(ne, avp, epi, dopa, phenyl)] = action_id
        
        self.valid_actions = list(self.action_table.keys())
        self.action_to_index_map = {k: k for k in self.valid_actions}
        self.n_actions = len(self.valid_actions)
    
    def decode(self, action: int) -> Dict[str, int]:
        """Return dose dict for a given action ID."""
        action = int(action)
        return self.action_table.get(action, self.action_table.get(0, {}))
    
    def action_to_idx(self, action: int) -> int:
        """Identity mapping (action ID == index)."""
        return int(action)
    
    def idx_to_action(self, idx: int) -> int:
        """Identity mapping."""
        return int(idx)
    
    def is_action_valid(self, action: int) -> bool:
        """Check if action ID exists in the action space."""
        return action in self.action_table
    
    def get_dose_description(self, doses: Dict[str, int]) -> str:
        """Convert dose dict to human-readable string."""
        parts = []
        order = ['norad_max', 'vaso_max', 'epi_max', 'dopa_max', 'phenyl_max']
        
        for drug in order:
            level = doses.get(drug, 0)
            if level > 0:
                name = Config.DRUG_MAPPING.get(drug, drug)
                if drug == 'dopa_max':
                    parts.append(f"{name}:Used")
                else:
                    parts.append(f"{name}:L{level}")
        
        return "; ".join(parts) if parts else "No drugs"

    def _get_rule_based_combos(self) -> List[Tuple[int, ...]]:
        """Enumerate all drug combinations that satisfy clinical constraints."""
        combos: Set[Tuple[int, ...]] = set()
        
        def add_combo(ne: int, avp: int = 0, epi: int = 0, dopa: int = 0, phe: int = 0) -> None:
            combo = apply_clinical_constraints(ne, avp, epi, dopa, phe)
            combos.add(combo)
        
        # NE = L0
        add_combo(0)
        for phe in [1, 2, 3]:
            add_combo(0, phe=phe)
        for avp in [1, 2, 3]:
            add_combo(0, avp=avp)
            for phe in [1, 2, 3]:
                add_combo(0, avp=avp, phe=phe)
        for epi in [1, 2, 3]:
            add_combo(0, epi=epi)
            for phe in [1, 2, 3]:
                add_combo(0, epi=epi, phe=phe)
        for dopa in [1]:
            add_combo(0, dopa=dopa)

        # NE = L1
        add_combo(1)
        for phe in [1, 2]:
            add_combo(1, phe=phe)
        for avp in [1, 2]:
            for phe in [0, 1, 2]:
                add_combo(1, avp=avp, phe=phe)
        for epi in [1, 2]:
            for phe in [0, 1, 2]:
                add_combo(1, epi=epi, phe=phe)

        # NE = L2
        add_combo(2)
        add_combo(2, dopa=1)
        for phe in [1, 2]:
            add_combo(2, phe=phe)
        for avp in [1, 2, 3]:
            for dopa in [0, 1]:
                for phe in [0, 1, 2]:
                    add_combo(2, avp=avp, dopa=dopa, phe=phe)
        for epi in [1, 2, 3]:
            for dopa in [0, 1]:
                for phe in [0, 1, 2]:
                    add_combo(2, epi=epi, dopa=dopa, phe=phe)

        # NE = L3+
        add_combo(3)
        for phe in [1, 2, 3]:
            add_combo(3, phe=phe)
        for avp in [1, 2, 3]:
            add_combo(3, avp=avp)
            for phe in [1, 2, 3]:
                add_combo(3, avp=avp, phe=phe)
        for dopa in [1, 2]:
            add_combo(3, dopa=dopa)
            for phe in [1, 2, 3]:
                add_combo(3, dopa=dopa, phe=phe)
            for avp in [1, 2, 3]:
                add_combo(3, avp=avp, dopa=dopa)

        return list(combos)


class DataProcessor:
    def __init__(self, logger=None, high_freq_combos=None):
        self.logger = logger
        self.scalers: Dict[str, StandardScaler] = {}
        self.physician_policy: Optional[Dict[int, Dict[int, float]]] = None
        self.samples: List[Dict[str, Any]] = []
        self.pid_to_samples: Dict[int, List[int]] = defaultdict(list)
        self.action_decoder = ActionDecoder(logger=self.logger, drug_mapping=Config.DRUG_MAPPING, high_freq_combos=high_freq_combos)
        self.valid_actions = self.action_decoder.valid_actions
        self.state_features: List[str] = []
        if self.logger:
            self._log_valid_action_examples()
    
    def _log_valid_action_examples(self) -> None:
        """Log example actions from the final action space."""
        self.logger.info(f"Final action space contains {len(self.valid_actions)} actions.")
        count = len(self.valid_actions)
        indices = sorted(set([0, 1, count//2, count-1]))
        for a in indices:
            if a < count:
                doses = self.action_decoder.decode(a)
                desc = self.action_decoder.get_dose_description(doses)
                self.logger.info(f"  Action {a}: {desc}")

    def _map_row_to_action(self, row: pd.Series) -> Tuple[int, bool, Tuple[int, int, int, int, int]]:
        """
        Map a row of raw drug doses to an action ID.
        Returns: (action_id, is_missed, discretized_tuple)
        """
        ne = discretize(row.get('norad_max', 0.0), 'norad_max')
        avp = discretize(row.get('vaso_max', 0.0), 'vaso_max')
        epi = discretize(row.get('epi_max', 0.0), 'epi_max')
        dopa = discretize(row.get('dopa_max', 0.0), 'dopa_max')
        phe = discretize(row.get('phenyl_max', 0.0), 'phenyl_max')
        key = apply_clinical_constraints(ne, avp, epi, dopa, phe)
        
        if key in self.action_decoder.dose_tuple_to_action:
            action_id = self.action_decoder.dose_tuple_to_action[key]
            return action_id, False, key
        else:
            return 0, True, key  # fallback to action 0

    def _discretize_actions(self, df: pd.DataFrame) -> pd.Series:
        """Apply discretization and mapping to entire DataFrame."""
        actions = []
        drug_cols = list(Config.DRUG_MAPPING.keys())
        for col in drug_cols:
            if col not in df.columns:
                df[col] = 0
        for _, row in df[drug_cols].iterrows():
            actions.append(self._map_row_to_action(row))
        return pd.Series(actions, index=df.index)

    def calculate_reward(
        self, 
        is_terminal: bool, 
        current: Dict[str, float], 
        next_state_dict: Optional[Dict[str, float]],
        action: int
    ) -> float:
        """Compute reward based on clinical outcomes and action intensity."""
        if is_terminal:
            return 15.0 if current.get('mortality', 0) == 0 else -15.0

        doses = self.action_decoder.decode(action)
        ne_lvl = doses.get('norad_max', 0)
        total_intensity = sum(v for k, v in doses.items() if v > 0)
        is_dosing = ne_lvl > 0
        next_map = next_state_dict.get('mean_bp', current.get('mean_bp', 0)) if next_state_dict else current.get('mean_bp', 0)
        
        reward = 0.0
        if next_map < 65:
            if is_dosing:
                reward += 2.0 * ne_lvl + (1.0 if total_intensity > ne_lvl else 0)
            else:
                reward -= 2.0
        elif next_map >= 72:
            if not is_dosing:
                reward += 2.0
            else:
                reward -= 1.0 * total_intensity
        else:
            reward += 0.5 if is_dosing else -0.5
        
        if total_intensity > 5:
            reward -= 1.0
        
        return reward
    
    def _add_missing_drugs(self, df: pd.DataFrame) -> None:
        """Ensure all drug columns exist (fill with 0 if missing)."""
        for drug in Config.FEATURE_GROUPS['drugs']:
            if drug not in df.columns:
                df[drug] = 0.0

    def _preprocess_all_features(self, df):
        """Preprocess and standardize features by group."""
        # Save raw HR before normalization
        for hr_col in ['HR']:
            if hr_col in df.columns:
                df[f'{hr_col}_raw'] = df[hr_col].copy()
        
        features_ordered = []
        # Binary features (e.g., gender)
        for col in Config.FEATURE_GROUPS['binary']:
            if col in df.columns:
                df[col] = df[col].fillna(0).clip(0, 1).astype(int) - 0.5
                features_ordered.append(col)
        # Normally distributed features
        for col in Config.FEATURE_GROUPS['normal']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
                if col in ['MeanBP', 'SysBP', 'DiaBP']:
                    df[col] = df[col] / 100.0  # Normalize BP to ~[0, 2]
                else:
                    scaler = StandardScaler()
                    data_vec = df[col].values.reshape(-1, 1)
                    df[col] = scaler.fit_transform(data_vec).flatten()
                    self.scalers[col] = scaler
                features_ordered.append(col)
        # Log-transformed features
        for col in Config.FEATURE_GROUPS['log']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
                min_val = df[col].min()
                if min_val <= 0:
                    df[col] = df[col] - min_val + 1e-6
                df[col] = np.log1p(df[col])
                scaler = StandardScaler()
                data_vec = df[col].values.reshape(-1, 1)
                df[col] = scaler.fit_transform(data_vec).flatten()
                self.scalers[col] = scaler
                features_ordered.append(col)
        
        self.state_features = features_ordered

    def _create_samples(self, df):
        """Create transition samples with filtering and reward calculation."""
        sample_id = 0
        kept_samples = 0
        total_samples = 0
        total_mapped = 0      
        missed_mapping = 0    
        group_col = 'icustayid' 
        
        for pid, pid_df in df.groupby(group_col):
            patient_outcome = pid_df.iloc[0].get('mortality_90d', 0)
            
            pid_actions = []
            pid_keys = []
            pid_missed_flags = []
            
            for _, row in pid_df.iterrows():
                action, is_missed, key = self._map_row_to_action(row)
                pid_actions.append(action)
                pid_keys.append(key)
                pid_missed_flags.append(is_missed)
                total_mapped += 1
                if is_missed:
                    missed_mapping += 1
            
            pid_actions = np.array(pid_actions)
            pid_keys = np.array(pid_keys)
            pid_missed_flags = np.array(pid_missed_flags)
            
            for idx in range(len(pid_df)):
                total_samples += 1
                row = pid_df.iloc[idx]
                
                raw_hr = row.get('HR_raw', row.get('HR', 0))
                has_drug = any(row.get(col, 0) > 0 for col in Config.FEATURE_GROUPS['drugs'])
                mean_bp_norm = row.get('MeanBP', 0.8)
                is_critical = (mean_bp_norm < 0.72) or (row.get('SOFA', 0) > 5.0)
                is_tachycardia = bool(raw_hr >= 100)

                # Keep only clinically relevant samples
                if not (has_drug or is_critical or is_tachycardia):
                    continue
                
                state_vector = self._get_state_vector(row)
                action = int(pid_actions[idx])
                key = tuple(pid_keys[idx])
                is_missed = pid_missed_flags[idx]
                
                reward_features = {
                    'mortality': patient_outcome,
                    'sofa': row.get('SOFA', 0),
                    'mean_bp': mean_bp_norm * 100.0,
                    'heart_rate': raw_hr
                }
                                
                raw_clinical = {
                    'norad_max': row.get('norad_max', 0.0),
                    'vaso_max': row.get('vaso_max', 0.0),
                    'epi_max': row.get('epi_max', 0.0),
                    'dopa_max': row.get('dopa_max', 0.0),
                    'phenyl_max': row.get('phenyl_max', 0.0),
                    'mean_bp': row.get('MeanBP', 0) * 100.0  
                }

                next_idx = idx + 1
                next_state_vector = None
                next_reward_features = {}
                done = False
                next_action = None
                if next_idx < len(pid_df):
                    next_action = int(pid_actions[next_idx])
                    next_row = pid_df.iloc[next_idx]
                    next_state_vector = self._get_state_vector(next_row)
                    next_reward_features = {
                        'sofa': next_row.get('SOFA', 0),
                        'mean_bp': next_row.get('MeanBP', 0.8) * 100.0
                    }
                else:
                    done = True

                self.samples.append({
                    'pid': pid,
                    'icustayid': pid,
                    'timestep': idx,
                    'state': state_vector,
                    'next_state': next_state_vector,
                    'action': action,
                    'next_action': next_action,
                    'action_idx': action,
                    'current_clinical': reward_features,
                    'current_clinical_raw': raw_clinical,
                    'next_clinical': next_reward_features,
                    'done': done,
                    'discretized_action_tuple': key,          
                    'is_action_mapped': not is_missed,        
                    'outcome': patient_outcome
                })
                
                self.pid_to_samples[pid].append(sample_id)
                sample_id += 1
                kept_samples += 1
        
        ratio = (kept_samples / total_samples * 100) if total_samples > 0 else 0
        mapping_failure_rate = (missed_mapping / total_mapped * 100) if total_mapped > 0 else 0
        
        if self.logger:
            self.logger.info(f"Sample filtering done: kept {kept_samples}/{total_samples} ({ratio:.1f}%)")
            self.logger.info(f"Action mapping failure rate: {missed_mapping}/{total_mapped} ({mapping_failure_rate:.2f}%)")

    def _get_state_vector(self, row: pd.Series) -> np.ndarray:
        """Extract and format state vector from a data row."""
        state_vector = []
        for col in self.state_features:
            if col in row and not pd.isna(row[col]):
                state_vector.append(row[col])
            else:
                state_vector.append(0.0)
        return np.array(state_vector, dtype=np.float32)
    
    def _build_physician_policy(self) -> None:
        """Estimate physician policy by state-action frequencies."""
        state_actions = defaultdict(lambda: defaultdict(int))
        for s in self.samples:
            state_hash = hash(tuple(s['state'][:Config.STATE_VECTOR_TRUNC_LEN])) % Config.STATE_HASH_MOD
            state_actions[state_hash][s['action']] += 1
        
        policy = defaultdict(dict)
        for state, actions in state_actions.items():
            total = sum(actions.values())
            if total > 0:
                policy[state] = {a: c / total for a, c in actions.items()}
        policy['__default__'] = {0: 1.0}
        self.physician_policy = dict(policy)

    def preprocess(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Full preprocessing pipeline."""
        df = df.copy()
        if self.logger:
            self.logger.info("Starting feature preprocessing...")
        self._add_missing_drugs(df)
        self._preprocess_all_features(df) 
        if self.logger:
            self.logger.info("Generating and filtering samples...")
        self._create_samples(df)
        if self.logger:
            self.logger.info("Building physician policy...")
        self._build_physician_policy()
        return self.samples

    def split_data(self, val_split: float = 0.1) -> Dict[str, Any]:
        """Split data into train/val/test sets with optional oversampling."""
        np.random.seed(Config.RANDOM_SEED)
        pids = list(self.pid_to_samples.keys())
        np.random.shuffle(pids)
        total = len(pids)
        train_size = int(0.8 * total)
        val_size = int(val_split * total)
        
        train_pids = set(pids[:train_size])
        val_pids = set(pids[train_size:train_size+val_size])
        test_pids = set(pids[train_size+val_size:])
        
        train_raw = [s for s in self.samples if s['pid'] in train_pids]
        val = [s for s in self.samples if s['pid'] in val_pids]
        test = [s for s in self.samples if s['pid'] in test_pids]
        
        train_final = []
        oversampling_ratio = getattr(Config, 'OVERSAMPLING_RATIO', 1)
        if oversampling_ratio > 1:
            if self.logger:
                self.logger.info(f"Applying oversampling Ratio={oversampling_ratio}")
            no_drug = [s for s in train_raw if s['action'] == 0]
            drug = [s for s in train_raw if s['action'] > 0]
            train_final = no_drug + (drug * oversampling_ratio)
            np.random.shuffle(train_final)
        else:
            train_final = train_raw
            
        def to_trans(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Add computed rewards and pad missing next states."""
            res = []
            for s in samples:
                r = self.calculate_reward(s['done'], s['current_clinical'], s['next_clinical'], s['action'])
                s_copy = s.copy()
                s_copy['reward'] = r
                if s_copy['next_state'] is None:
                    s_copy['next_state'] = np.zeros_like(s_copy['state'], dtype=np.float32)
                res.append(s_copy)
            return res
        
        train_trans = to_trans(train_final)
        val_trans = to_trans(val)
        test_trans = to_trans(test)
        
        if self.logger:
            self.logger.info(f"Data split complete: train {len(train_trans)}, val {len(val_trans)}, test {len(test_trans)}")
        
        return {
            'train_trans': train_trans,
            'val': val_trans,
            'test': test_trans,
            'state_dim': len(self.state_features),
            'action_dim': self.action_decoder.n_actions,
            'valid_actions': self.valid_actions,
            'physician_policy': self.physician_policy,
            'state_features': self.state_features,
            'action_decoder': self.action_decoder
        }