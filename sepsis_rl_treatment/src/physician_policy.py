import numpy as np
import random
from collections import defaultdict

from config import Config

class PhysicianPolicy:
    def __init__(self, physician_policy, valid_actions, logger=None):
        self.logger = logger
        self.physician_policy = physician_policy
        self.valid_actions = valid_actions
        self.action_distribution = self._compute_global_dist()
    
    def _compute_global_dist(self):
        counts = defaultdict(float)
        for state, dist in self.physician_policy.items():
            if state == '__default__' or not isinstance(dist, dict): continue
            total = sum(dist.values()) + 1e-10
            for a, freq in dist.items():
                if a in self.valid_actions:  
                    counts[a] += freq / total
        total = sum(counts.values()) + 1e-10
        return {a: c / total for a, c in counts.items()}
    
    def _get_state_distribution(self, state):
        if isinstance(state, (np.ndarray, list)):
            state_hash = hash(tuple(state[:Config.STATE_VECTOR_TRUNC_LEN])) % Config.STATE_HASH_MOD  
        else:
            state_hash = state
        raw_dist = self.physician_policy.get(state_hash, self.physician_policy.get('__default__', {}))
        valid_dist = {a: p for a, p in raw_dist.items() if a in self.valid_actions}
        if not valid_dist: return self.action_distribution
        total = sum(valid_dist.values()) + 1e-10
        return {a: p / total for a, p in valid_dist.items()}
    
    def get_action_prob(self, state, action):
        dist = self._get_state_distribution(state)
        return dist.get(action, 1e-10)
     
    def select_action(self, state, training=False, epsilon=None, deterministic=False):  
        action_dist = self._get_state_distribution(state)
        if not action_dist: return random.choice(self.valid_actions)
        
        if deterministic:
            max_prob = max(action_dist.values())
            best_actions = [a for a, p in action_dist.items() if p == max_prob]
            return random.choice(best_actions)
        else:
            actions = list(action_dist.keys())
            probs = np.array(list(action_dist.values()))
            probs = probs / (probs.sum() + 1e-10)
            return np.random.choice(actions, p=probs)
    
    def evaluate(self, data_processor, val_samples):
        rewards = []
        for sample in val_samples:
            if sample['next_clinical'] is None:
                continue
            action = sample['action'] 
            reward = data_processor.calculate_reward(
                sample['done'], sample['current_clinical'], sample['next_clinical'], action
            )
            rewards.append(reward)
        
        avg_reward = np.mean(rewards) if rewards else 0.0
        std_reward = np.std(rewards) if rewards else 0.0
        
        return avg_reward, std_reward