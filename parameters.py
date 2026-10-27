import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
BATCH_SIZE = 256
LEARNING_RATE = 0.0003
EPOCH = 30

# Data
WINDOW_SIZE = 5
PREDICT_SIZE = 5
LABEL_SIZE = 5

TRAIN_NUM_SAMPLES=1005
NUM_SAMPLES = 1005
VELOCITY = 20
TRAIN_VEL = 1035
dt= 0.05
MT = 32
TRACKING_TIME = 50
NY = 32
NZ = 32
C=3e8
NSC = 256
FC = 28e9
BANDWIDTH = 400e6
P=10
SNR_DB = 100
MAX_NUM_BEAMS = 4

DELAY_FACTOR = 1
NUM_CLASSES = MT*MT+1
CODEBOOK = 'DFT'

# Transformer
FEATURE_SIZE = 256
N_HEAD = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 2
DROPOUT = 0.1


# MODEL_PATH = f"./model state/beam_prediction_model_vel{TRAIN_VEL}_mt{MT}_{CODEBOOK}_small.pth"
MODEL_PATH = f"./model state/beam_prediction_model_vel{TRAIN_VEL}_mt{MT}_{CODEBOOK}_newloss.pth"
DATA_PATH = f"./data/training_data_vel{TRAIN_VEL}_mt{MT}_dft.mat"
TESTDATA_PATH = f"./data/testing_data_vel{VELOCITY}_mt{MT}_dft.mat"
RE_DATA_PATH = f"./data/training_data.mat"
RESULT_PATH = f"./results/tracking_results_vel{VELOCITY}_mt{MT}_{CODEBOOK}.mat"

## REFINE MODEL PARAMETERS
# MODEL_PATH_R = f"./model state/reacquisition_model_mt{MT}_small.pth"
MODEL_PATH_R = f"./model state/reacquisition_model_mt{MT}_newloss.pth"
DATA_PATH_R = f"./data/target_lost_data_mt{MT}_newloss.mat"
RESULT_PATH_R = f"./results/refine_result_vel{VELOCITY}_mt{MT}.mat"
FEATURE_SIZE_R = 128
BATCH_SIZE_R = 32
