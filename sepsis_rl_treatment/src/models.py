import numpy as np
import os
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import itertools 
import time

from config import Config


class AuxTaskMonitor:
    """Monitors auxiliary task (mortality prediction) performance and dynamically adjusts its loss weight."""
    def __init__(self, initial_weight=Config.AUX_LOSS_WEIGHT):
        self.current_weight = initial_weight
        self.best_auc = 0.0
        self.auc_history = []
    
    def update(self, pred_logits, labels):
        """Update AUC and adjust weight based on current performance."""
        pred_probs = torch.sigmoid(pred_logits).detach().cpu().numpy()  # ✅ fixed
        labels = labels.detach().cpu().numpy()
        try:
            auc = roc_auc_score(labels, pred_probs) if len(np.unique(labels)) > 1 else 0.5
        except:
            auc = 0.5
        self.auc_history.append(auc)
        
        # Dynamically reduce weight as AUC improves
        if auc >= 0.85:
            self.current_weight = 0.1
        elif auc >= 0.75:
            self.current_weight = 0.2
        elif auc >= 0.65:
            self.current_weight = 0.3
        else:
            self.current_weight = 0.5  
        self.best_auc = max(self.best_auc, auc)
        return auc
        
    def get_weight(self):
        return self.current_weight


class MTLBaseModule(nn.Module):
    """Base class for multi-task RL models with shared encoder."""
    def __init__(self):
        super(MTLBaseModule, self).__init__()
        self.data_processor = None
        self.attention_weights = None
        
    def forward(self, state):
        raise NotImplementedError
        
    def get_target_q(self, state):
        q_values, _ = self.forward(state)
        return q_values
    
    def set_data_processor(self, data_processor):
        self.data_processor = data_processor

    def get_shared_parameters(self):
        raise NotImplementedError

    def select_action(self, state, epsilon=Config.EVAL_EPSILON, training=False, 
                      deterministic=False, temperature=1.0):
        """Select action using valid action masking and optional exploration."""
        if self.data_processor is None:
            # Fallback (should not occur in practice)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
                q_values, _ = self.forward(state_tensor)
                return q_values.argmax().item(), None

        # Restrict to valid actions only
        valid_action_idxs = [idx for idx, action in enumerate(self.data_processor.valid_actions)
                             if self.data_processor.action_decoder.is_action_valid(action)]
        if not valid_action_idxs:
            valid_action_idxs = list(range(len(self.data_processor.valid_actions)))

        if deterministic:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
                q_values, _ = self.forward(state_tensor)
                valid_q_values = q_values[:, valid_action_idxs]
                best_valid_idx = valid_q_values.argmax().item()
                return valid_action_idxs[best_valid_idx], None
        else:
            if random.random() < epsilon:
                return random.choice(valid_action_idxs), None
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
                    q_values, _ = self.forward(state_tensor)
                    q_values = q_values.squeeze()
                    valid_q_values = q_values[valid_action_idxs]
                    action_probs = F.softmax(valid_q_values / temperature, dim=0).cpu().numpy()
                    action_probs /= action_probs.sum()  # normalize for numerical stability
                    valid_action_internal_idx = np.random.choice(len(valid_action_idxs), p=action_probs)
                    return valid_action_idxs[valid_action_internal_idx], action_probs


class MTLDuelingQLearning(MTLBaseModule):
    """Multi-task Dueling DQN with separate value and advantage streams."""
    def __init__(self, state_dim, action_dim):
        super(MTLDuelingQLearning, self).__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, action_dim)
        )
        self.outcome_head = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM // 2, 1)
        )
        
        # Initialize target networks
        self.target_shared_encoder = copy.deepcopy(self.shared_encoder)
        self.target_value_stream = copy.deepcopy(self.value_stream)
        self.target_advantage_stream = copy.deepcopy(self.advantage_stream)
        self._set_target_eval()

    def _set_target_eval(self):
        """Set target networks to eval mode (no dropout, no batch norm updates)."""
        self.target_shared_encoder.eval()
        self.target_value_stream.eval()
        self.target_advantage_stream.eval()
        
    def get_shared_parameters(self):
        return self.shared_encoder.parameters()

    def forward(self, state):
        features = self.shared_encoder(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        outcome_logits = self.outcome_head(features)
        return q_values, outcome_logits

    def get_target_q(self, state):
        with torch.no_grad():
            features = self.target_shared_encoder(state)
            value = self.target_value_stream(features)
            advantage = self.target_advantage_stream(features)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values
            
    def update_target_network(self, tau=Config.TAU):
        """Soft update target networks."""
        for target_param, param in zip(self.target_shared_encoder.parameters(), self.shared_encoder.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_value_stream.parameters(), self.value_stream.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_advantage_stream.parameters(), self.advantage_stream.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def compute_cql_loss(self, states, actions, q_values):
        """Conservative Q-Learning loss to penalize OOD actions."""
        logsumexp = torch.logsumexp(q_values, dim=1).mean()
        q_at_actions = q_values.gather(1, actions.unsqueeze(1)).mean()
        return Config.CQL_ALPHA * (logsumexp - q_at_actions)


class AttentionLayer(nn.Module):
    """Multi-head self-attention with residual connections and feed-forward layer."""
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.num_heads = Config.NUM_HEADS
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=self.num_heads, 
            batch_first=True,
            dropout=0.1
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.1)
        )
        self.dropout = nn.Dropout(0.1)
        self.last_attn_weights = None

    def forward(self, x):
        # Handle older PyTorch versions
        try:
            attn_output, attn_weights = self.attention(
                x, x, x, need_weights=True, average_attn_weights=False 
            )
        except TypeError:
            attn_output, attn_weights = self.attention(x, x, x, need_weights=True)
            if attn_weights.dim() == 3:
                attn_weights = attn_weights.unsqueeze(1)
        self.last_attn_weights = attn_weights
        x = self.layer_norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x, attn_weights

    def get_attention_weights(self):
        return self.last_attn_weights
    

class MTLAttentionCQL(MTLBaseModule):
    """MTL model with attention-based state encoder and CQL regularization."""
    def __init__(self, state_dim, action_dim):
        super(MTLAttentionCQL, self).__init__()
        self.alpha = Config.CQL_ALPHA 
        self.num_tokens = getattr(Config, 'NUM_TOKENS', 4) 
        
        # Tokenization layer
        self.tokenizer = nn.Sequential(
            nn.Linear(state_dim, Config.HIDDEN_DIM * self.num_tokens),
            nn.LayerNorm(Config.HIDDEN_DIM * self.num_tokens),
            nn.ReLU()
        )
        self.shared_attention = AttentionLayer(Config.HIDDEN_DIM)
        
        self.value_stream = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, action_dim)
        )
        self.outcome_head = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM // 2, 1)
        )
        
        # Target networks
        self.target_tokenizer = copy.deepcopy(self.tokenizer)
        self.target_shared_attention = copy.deepcopy(self.shared_attention)
        self.target_value_stream = copy.deepcopy(self.value_stream)
        self.target_advantage_stream = copy.deepcopy(self.advantage_stream)
        self._set_target_eval()

    def _set_target_eval(self):
        self.target_tokenizer.eval()
        self.target_shared_attention.eval()
        self.target_value_stream.eval()
        self.target_advantage_stream.eval()

    def get_shared_parameters(self):
        return itertools.chain(self.tokenizer.parameters(), self.shared_attention.parameters())

    def _get_features(self, state, use_target=False):
        """Encode state into a fixed-length representation using attention."""
        token_net = self.target_tokenizer if use_target else self.tokenizer
        attn_net = self.target_shared_attention if use_target else self.shared_attention
        
        batch_size = state.size(0)
        flat_tokens = token_net(state)
        x_seq = flat_tokens.view(batch_size, self.num_tokens, Config.HIDDEN_DIM)
        
        attn_output, attn_weights = attn_net(x_seq)
        if not use_target:
            self.attention_weights = attn_weights 
        features = attn_output.mean(dim=1)
        return features

    def forward(self, state):
        features = self._get_features(state, use_target=False)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        outcome_logits = self.outcome_head(features)
        return q_values, outcome_logits

    def get_target_q(self, state):
        with torch.no_grad():
            features = self._get_features(state, use_target=True)
            value = self.target_value_stream(features)
            advantage = self.target_advantage_stream(features)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values

    def update_target_network(self, tau=Config.TAU):
        """Soft update all target network components."""
        for local_mod, target_mod in [
            (self.tokenizer, self.target_tokenizer),  
            (self.shared_attention, self.target_shared_attention),
            (self.value_stream, self.target_value_stream),
            (self.advantage_stream, self.target_advantage_stream)
        ]:
            for target_param, param in zip(target_mod.parameters(), local_mod.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def compute_cql_loss(self, states, actions, q_values):
        logsumexp = torch.logsumexp(q_values, dim=1).mean()
        q_at_actions = q_values.gather(1, actions.unsqueeze(1)).mean()
        return self.alpha * (logsumexp - q_at_actions)
    
    def get_attention_weights(self):
        if self.attention_weights is not None:
            return self.attention_weights.detach()
        return None
        

class MTL_RLTrainer:
    """Trainer for multi-task RL agents with optional CQL and behavior cloning."""
    def __init__(self, model, optimizer, logger, data_processor, device=Config.DEVICE, 
                 enable_mtl=True, enable_cql=True):
        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.device = device
        self.data_processor = data_processor 
        self.enable_mtl = enable_mtl
        self.enable_cql = enable_cql
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.DECAY_STEP, gamma=Config.LEARNING_RATE_DECAY)
        self.best_val_reward = -float('inf')
        self.aux_criterion = nn.BCEWithLogitsLoss()
        self.aux_monitor = AuxTaskMonitor()
        self.aux_auc_history = []
        
        # Behavior cloning loss with class weighting (down-weight "no drug" action)
        action_dim = len(data_processor.valid_actions)
        class_weights = torch.ones(action_dim).to(device)
        
        try:
            no_drug_val = 0
            if no_drug_val in data_processor.valid_actions:
                no_drug_idx = data_processor.valid_actions.index(no_drug_val)
                class_weights[no_drug_idx] = 0.05 
                logger.info(f"BC Loss: Action 0 (Index {no_drug_idx}) weight = 0.05")
            else:
                logger.warning("Action 0 not found; using uniform weights")
        except Exception as e:
            logger.warning(f"Error setting class weights: {e}")
            
        self.bc_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.bc_weight = 5.0 
    
    def create_dataloader(self, transitions, batch_size=Config.BATCH_SIZE, shuffle=True):
        """Convert list of transitions into a PyTorch DataLoader."""
        states = torch.FloatTensor(np.array([t['state'] for t in transitions])).to(self.device)
        actions_idx = [self.data_processor.action_decoder.action_to_idx(t['action']) for t in transitions]
        actions = torch.LongTensor(actions_idx).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in transitions]).to(self.device)
        next_states = torch.FloatTensor(np.array([t['next_state'] for t in transitions])).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in transitions]).to(self.device)
        outcomes = torch.FloatTensor([t['outcome'] for t in transitions]).unsqueeze(1).to(self.device)
        return DataLoader(TensorDataset(states, actions, rewards, next_states, dones, outcomes), batch_size=batch_size, shuffle=shuffle)
        
    def train_step(self, batch, gamma=Config.GAMMA):
        """Single training step with RL, BC, CQL, and optional auxiliary task."""
        states, actions, rewards, next_states, dones, outcomes = batch
        
        # Forward pass
        current_q_all, outcome_logits = self.model(states)
        current_q = current_q_all.gather(1, actions.unsqueeze(1))
        
        # TD loss
        with torch.no_grad():
            next_q = self.model.get_target_q(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards.unsqueeze(1) + gamma * next_q * (1 - dones.unsqueeze(1))
        loss_rl = F.mse_loss(current_q, target_q)
        
        # CQL regularization (penalize overestimation on OOD actions)
        if self.enable_cql and hasattr(self.model, 'compute_cql_loss') and Config.CQL_ALPHA > 0:
            loss_rl += self.model.compute_cql_loss(states, actions, current_q_all)

        # Weighted behavior cloning loss
        loss_bc = self.bc_criterion(current_q_all, actions)
        total_main_loss = loss_rl + (self.bc_weight * loss_bc)

        aux_auc = 0.5
        self.optimizer.zero_grad()
        
        if not self.enable_mtl:
            total_main_loss.backward()
            self.optimizer.step()
            return total_main_loss.item(), aux_auc

        # Multi-task learning: add auxiliary mortality prediction loss
        loss_aux = self.aux_criterion(outcome_logits, outcomes)
        dynamic_aux_weight = self.aux_monitor.get_weight()
        aux_auc = self.aux_monitor.update(outcome_logits, outcomes)
        loss_mtl_aux = dynamic_aux_weight * loss_aux
        
        final_loss = total_main_loss + loss_mtl_aux
        final_loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.GRADIENT_CLIP)
        self.optimizer.step()
        
        return final_loss.item(), aux_auc
    
    def train_epoch(self, dataloader, gamma=Config.GAMMA):
        """Train one full epoch."""
        self.model.train()
        total_loss = 0
        total_aux_auc = 0
        for batch in dataloader:
            loss, aux_auc = self.train_step(batch, gamma)
            total_loss += loss
            total_aux_auc += aux_auc
        avg_loss = total_loss / len(dataloader)
        avg_aux_auc = total_aux_auc / len(dataloader) if self.enable_mtl else 0.5
        self.aux_auc_history.append(avg_aux_auc)
        return avg_loss

    def evaluate(self, val_samples):
        """Evaluate policy on validation set using deterministic action selection."""
        self.model.eval()
        rewards = []
        with torch.no_grad():
            for sample in val_samples:
                if sample['next_clinical'] is None: 
                    continue
                action_idx, _ = self.model.select_action(
                    sample['state'], epsilon=0.0, training=False, deterministic=True
                )
                action = self.data_processor.action_decoder.idx_to_action(action_idx)
                
                reward = self.data_processor.calculate_reward(
                    sample['done'], sample['current_clinical'], sample['next_clinical'], action
                )
                rewards.append(reward)
        if not rewards:
            return 0.0, 0.0
        return np.mean(rewards), np.std(rewards)

    def train(self, train_trans, val_samples, epochs=Config.TRAIN_EPOCHS, save_best=True):
        """Full training loop with logging and learning rate scheduling."""
        train_loader = self.create_dataloader(train_trans)
        history = {'train_loss': [], 'val_reward': [], 'aux_auc': [], 'aux_weight': [], 'lr': [], 'bc_weight': []}
        start_time = time.time()
        
        # Behavior cloning weight decay (start strong, decay to maintain baseline constraint)
        bc_decay = 0.98 
        min_bc_weight = 2.0 
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.model.update_target_network()
            val_reward_mean, val_reward_std = self.evaluate(val_samples)
            
            history['train_loss'].append(train_loss)
            history['val_reward'].append(val_reward_mean)
            history['aux_auc'].append(self.aux_auc_history[-1] if self.aux_auc_history else 0)
            history['aux_weight'].append(self.aux_monitor.get_weight())
            history['lr'].append(self.scheduler.get_last_lr()[0])
            history['bc_weight'].append(self.bc_weight)
            
            self.scheduler.step()
            
            # Decay BC weight but keep minimum constraint
            self.bc_weight = max(min_bc_weight, self.bc_weight * bc_decay)
            
            if save_best and val_reward_mean > self.best_val_reward:
                self.best_val_reward = val_reward_mean
            
            if (epoch + 1) % 5 == 0: 
                elapsed = time.time() - start_time
                self.logger.info(f"Ep {epoch+1} | Loss:{train_loss:.3f} | Val R:{val_reward_mean:.3f} | BC W:{self.bc_weight:.3f}")
                
        return history
