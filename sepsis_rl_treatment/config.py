import os
import torch

class Config:
    """Configuration class: retains NE, EPI, AVP, PHE"""

    # Data paths
    DATA_PATH = './dataset/mimictabl.csv'  # MIMIC-III dataset
    OUTPUT_PATH = './outputs/'
    OUTPUT_DATA_PATH = os.path.join(OUTPUT_PATH, "data")
    OUTPUT_MODEL_PATH = os.path.join(OUTPUT_PATH, "models")
    OUTPUT_FIG_PATH = os.path.join(OUTPUT_PATH, "figs")
    LOG_PATH = os.path.join(OUTPUT_PATH, "logs") 

    # Drug name mapping
    DRUG_MAPPING = {
        'norad_max': 'NE',   # μg/kg/min
        'vaso_max': 'AVP',   # U/min
        'epi_max': 'EPI',    # μg/kg/min
        'phenyl_max': 'PHE', # μg/kg/min
        'dopa_max': 'DOPA'
    }

    # Clinical dosage ranges (low, medium, high, max)
    CLINICAL_RANGES = {
        'norad_max': (0.0, 0.1, 0.5, float('inf')), 
        'vaso_max': (0.0, 0.05, 0.1, float('inf')),
        'epi_max': (0.0, 0.04, 0.06, float('inf')),
        'phenyl_max': (0.0, 2.5, 5.0, float('inf')),
        'dopa_max': (0.0, 20.0, float('inf'))
    }

    # Drug hierarchy
    FIRST_LINE_DRUG = 'norad_max' 
    SECOND_LINE_DRUGS = ['vaso_max', 'epi_max'] 
    AUXILIARY_DRUGS = ['phenyl_max', 'dopa_max'] 
    
    # Features used for reward calculation
    REWARD_FEATURES = ['mortality_90d', 'SOFA', 'Arterial_lactate', 'MeanBP']
    
    # Action space
    N_DRUGS = len(DRUG_MAPPING) 

    # Training settings
    RANDOM_SEED = 42
    GAMMA = 0.99                    # Discount factor
    TAU = 0.001                     # Soft update coefficient
    LEARNING_RATE_DECAY = 0.9       # Learning rate decay factor
    DECAY_STEP = 100                # Steps before decay
    TRAIN_EPOCHS = 20               # Total training epochs
    GRADIENT_CLIP = 1.0             # Gradient clipping threshold

    # Multi-task learning (MTL)
    AUX_LOSS_WEIGHT = 0.5           # Weight for auxiliary loss

    # CQL (Conservative Q-Learning) parameters
    CQL_ALPHA = 2.0                 # Conservative penalty weight
    CQL_DIVERSITY_PENALTY = 0.01    # Diversity penalty in CQL

    # Model architecture (reduced complexity)
    HIDDEN_DIM = 64                 # Hidden dimension size
    NUM_HEADS = 2                   # Number of attention heads
    NUM_TOKENS = 2                  # Number of tokens per drug

    # Exploration strategy (epsilon-greedy)
    EPSILON_MAX = 0.5               # Initial exploration rate
    EPSILON_MIN = 0.05              # Minimum exploration rate
    EPSILON_DECAY_RATE = 0.95       # Decay rate per episode
    EVAL_EPSILON = 0.05             # Exploration rate during evaluation

    # Training regularization
    BATCH_SIZE = 64                 # Batch size
    LEARNING_RATE = 5e-5            # Initial learning rate
    WEIGHT_DECAY = 1e-3             # L2 regularization strength
    ENTROPY_WEIGHT = 0.01           # Entropy regularization weight

    # Data balancing
    OVERSAMPLING_RATIO = 5          # Ratio for minority class oversampling

    # Off-policy evaluation (WIS)
    WIS_CLIP_MIN = 0.01             # Minimum weight clipping for WIS
    WIS_CLIP_MAX = 5.0              # Maximum weight clipping for WIS

    # Feature grouping for preprocessing
    FEATURE_GROUPS = {
        'binary': ['gender'],  
        'normal': [
            'age','Weight_kg','SysBP','DiaBP','MeanBP',  
            'RR','FiO2','Potassium','Sodium','Glucose',
            'Magnesium','Calcium','Hb','WBC_count','Platelets_count',
            'Arterial_pH','PaO2','paCO2','Arterial_BE','HCO3','PaO2_FiO2','SOFA',
            'HR', 'Chloride', 'Albumin', 'PTT', 'PT'
        ],  
        'log': ['SpO2','BUN','Creatinine','Total_bili','INR'],
        'drugs': list(DRUG_MAPPING.keys()) 
    }
    
    FULL_FEATURES = FEATURE_GROUPS['binary'] + FEATURE_GROUPS['normal'] + FEATURE_GROUPS['log'] + FEATURE_GROUPS['drugs']
    
    # State processing
    STATE_HASH_MOD = 1000           # Modulo for state hashing
    STATE_VECTOR_TRUNC_LEN = 5      # Truncate state history to last N steps

    # Device setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def create_output_dirs(cls):
        """Create output directories if they don't exist."""
        for path in [cls.OUTPUT_DATA_PATH, cls.OUTPUT_MODEL_PATH, cls.OUTPUT_FIG_PATH, cls.LOG_PATH]:
            os.makedirs(path, exist_ok=True)