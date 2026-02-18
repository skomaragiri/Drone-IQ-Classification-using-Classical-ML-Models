# DERIVATION PARAMETERS
NUM_FEATURES = 5
WINDOW_LEN = 4096 # measured in IQ samples
OVERLAP = 0.50


# DATA PARAMETERS
NUM_TRAINING_FILES = 25 # how many files in the saved numpy data for training
NUM_EVALUATION_FILES = 6 # how many files in the saved numpy data for evaluation
MAX_FILES = 1 # if not using the saved numpy data, this is the max files to intake
USE_SAVED_DATA = False # True = used the saved .npy file data instead of re-deriving the features again
SAVE_METRICS_TO_FILE = False

TRAINING_DATASET = ""
EVAL_DATASET = ""


TRAINING_DATASET = "n11_pro_comms_room_DL"
EVAL_DATASET = "n11_pro_comms_room_DL"


TRAINING_DATASET = "dji_10mhz_chamber"
EVAL_DATASET = "dji_10mhz_chamber"

# A very noisy dataset that usually has terrible results
TRAINING_DATASET = "dji_20mhz_comms"
EVAL_DATASET = "dji_20mhz_comms"



# DATASET PARAMETERS
# carlos' Mac
# data_dir = '/Users/carlos_1/Documents/GitHub/RFML-Code/RFML_Combined_Dataset_2025/RFML_Drone_Dataset_2025/old_drone_full_annotated_dataset/RFML_Old_Drone_Eval_data/*'
# uav-cyberlab-rfml laptop
training_data_dir = f'/home/uav-cyberlab-rfml/RFML/test-dataset/test_{TRAINING_DATASET}_training'
eval_data_dir = f'/home/uav-cyberlab-rfml/RFML/test-dataset/test_{TRAINING_DATASET}_eval'



FEATURES_TO_USE = [
    # TODO: FIGURE OUT HOW TO MAKE THE RETURN FUNCTION ACTUALLY UPDATE
        # checkout preprocessing.py: extractRFFeatures() and try to tie that function to this
    # "mean_I",
    # "mean_Q",
    "var_I",
    "var_Q",
    # "skew_I",
    # "kurt_I",
    # "skew_Q",
    # "kurt_Q",
    "mean_abs",
    # "var_abs",
    # "PAPR",
    # "corr_IQ",
    # "subbands",
    "total_power",
    "spectral_centroid", # you need centroid with bandwidth
    "spectral_bandwidth", # you need centroid with bandwidth
    # "spectral_flatness",
]