import os
import sys
from easydict import EasyDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

cfg = EasyDict()
_C = cfg

# test related setting  ------------------------------------------------------------------------------------------------
_C.TEST_ONLY = True
_C.WEIGHT = "../results/interpret_aud/11-06_00-35/checkpoint_21.pkl"
_C.COMPUTE_PCC = True
_C.COMPUTE_CCC = True

# data set split config ------------------------------------------------------------------------------------------------
_C.OUTPUT_DIR = "../results/interpret_aud"
_C.DATA_ROOT = "../datasets"

_C.TRAIN_AUD_DATA = "raw_voice/trainingData"
_C.VALID_AUD_DATA = "raw_voice/validationData"
_C.TEST_AUD_DATA = "raw_voice/testData"

_C.TRAIN_LABEL_DATA = "annotation/annotation_training.pkl"
_C.VALID_LABEL_DATA = "annotation/annotation_validation.pkl"
_C.TEST_LABEL_DATA = "annotation/annotation_test.pkl"

# data loader config ---------------------------------------------------------------------------------------------------
_C.TRAIN_BATCH_SIZE = 32  # 24
_C.VALID_BATCH_SIZE = 32  # 8
_C.SHUFFLE = True
_C.NUM_WORKERS = 4
_C.START_EPOCH = 0
_C.MAX_EPOCH = 100
# optimizer config -----------------------------------------------------------------------------------------------------
_C.LR_INIT = 0.05
_C.MOMENTUM = 0.9
_C.WEIGHT_DECAY = 0.0005
_C.FACTOR = 0.1
_C.MILESTONE = [50, 70, 90]

_C.RESUME = None

_C.LOG_INTERVAL = 10
