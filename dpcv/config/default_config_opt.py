import os
import os.path as osp
import numpy as np
from easydict import EasyDict as CfgNode

__C = CfgNode()
# Consumers can get config by:
cfg = __C
""" Follow commont steps to config a experiment:
step 1 : prepare dataset
step 2 : build a dataloader
step 3 : build a model
step 4 : set a loss function
step 5 : set a solver 
step 6 : build a trainer 
step 7 : set test metric
"""
# ------------------------------------------- step 1 : dataset config node ---------------------------------------------
__C.DATA = CfgNode()
__C.DATA.ROOT = "datasets"
__C.DATA.TYPE = "frame"
__C.DATA.SESSION = "talk"
# number of frames to sample from a video for each iteration
__C.DATA.SAMPLE_SIZE = 16
__C.DATA.SAMPLING_NAME = ""

__C.DATA.TRAIN_IMG_DATA = "image_data/train_data"
__C.DATA.TRAIN_IMG_FACE_DATA = "image_data/train_data_face"
__C.DATA.TRAIN_AUD_DATA = "raw_voice/trainingData"
__C.DATA.TRAIN_LABEL_DATA = "annotation/annotation_training.pkl"
__C.DATA.ANIMALS_TRAIN_IMG_DATA = "udiva_tiny/train/recordings/animals_recordings_train_img"
__C.DATA.GHOST_TRAIN_IMG_DATA = "udiva_tiny/train/recordings/ghost_recordings_train_img"
__C.DATA.LEGO_TRAIN_IMG_DATA = "udiva_tiny/train/recordings/lego_recordings_train_img"
__C.DATA.TALK_TRAIN_IMG_DATA = "udiva_tiny/train/recordings/talk_recordings_train_img"
__C.DATA.ANIMALS_TRAIN_AUD_DATA = "udiva_tiny/train/recordings/animals_recordings_train_wav"
__C.DATA.GHOST_TRAIN_AUD_DATA = "udiva_tiny/train/recordings/ghost_recordings_train_wav"
__C.DATA.LEGO_TRAIN_AUD_DATA = "udiva_tiny/train/recordings/lego_recordings_train_wav"
__C.DATA.TALK_TRAIN_AUD_DATA = "udiva_tiny/train/recordings/talk_recordings_train_wav"


__C.DATA.VALID_IMG_DATA = "image_data/valid_data"
__C.DATA.VALID_IMG_FACE_DATA = "image_data/valid_data_face"
__C.DATA.VALID_AUD_DATA = "raw_voice/validationData"
__C.DATA.VALID_LABEL_DATA = "annotation/annotation_validation.pkl"
__C.DATA.ANIMALS_VAL_IMG_DATA = "udiva_tiny/val/recordings/animals_recordings_val_img"
__C.DATA.GHOST_VAL_IMG_DATA = "udiva_tiny/val/recordings/ghost_recordings_val_img"
__C.DATA.LEGO_VAL_IMG_DATA = "udiva_tiny/val/recordings/lego_recordings_val_img"
__C.DATA.TALK_VAL_IMG_DATA = "udiva_tiny/val/recordings/talk_recordings_val_img"
__C.DATA.ANIMALS_VAL_AUD_DATA = "udiva_tiny/val/recordings/animals_recordings_val_wav"
__C.DATA.GHOST_VAL_AUD_DATA = "udiva_tiny/val/recordings/ghost_recordings_val_wav"
__C.DATA.LEGO_VAL_AUD_DATA = "udiva_tiny/val/recordings/lego_recordings_val_wav"
__C.DATA.TALK_VAL_AUD_DATA = "udiva_tiny/val/recordings/talk_recordings_val_wav"


__C.DATA.TEST_IMG_DATA = "image_data/test_data"
__C.DATA.TEST_IMG_FACE_DATA = "image_data/test_data_face"
__C.DATA.TEST_AUD_DATA = "raw_voice/testData"
__C.DATA.TEST_LABEL_DATA = "annotation/annotation_test.pkl"
__C.DATA.ANIMALS_TEST_IMG_DATA = "udiva_tiny/test/recordings/animals_recordings_test_img"
__C.DATA.GHOST_TEST_IMG_DATA = "udiva_tiny/test/recordings/ghost_recordings_test_img"
__C.DATA.LEGO_TEST_IMG_DATA = "udiva_tiny/test/recordings/lego_recordings_test_img"
__C.DATA.TALK_TEST_IMG_DATA = "udiva_tiny/test/recordings/talk_recordings_test_img"
__C.DATA.ANIMALS_TEST_AUD_DATA = "udiva_tiny/test/recordings/animals_recordings_test_wav"
__C.DATA.GHOST_TEST_AUD_DATA = "udiva_tiny/test/recordings/ghost_recordings_test_wav"
__C.DATA.LEGO_TEST_AUD_DATA = "udiva_tiny/test/recordings/lego_recordings_test_wav"
__C.DATA.TALK_TEST_AUD_DATA = "udiva_tiny/test/recordings/talk_recordings_test_wav"

__C.DATA.VA_ROOT = "datasets"
__C.DATA.VA_DATA = "va_data/cropped_aligned"
__C.DATA.VA_TRAIN_LABEL = "va_data/va_label/VA_Set/Train_Set"
__C.DATA.VA_VALID_LABEL = "va_data/va_label/VA_Set/Validation_Set"

# NoXI dataset
__C.DATA.NOXI_IMG_DATA = "noxi/img"
__C.DATA.NOXI_AUD_DATA = "noxi/wav"
__C.DATA.NOXI_LABEL_DATA = "noxi/session_label.pkl"

# ------------------------------------------ step 2 : dataloader config node -------------------------------------------
__C.DATA_LOADER = CfgNode()
# name of dataloader build function
__C.DATA_LOADER.NAME = "single_frame_data_loader"
__C.DATA_LOADER.DATASET = ""
__C.DATA_LOADER.TRANSFORM = "standard_frame_transform"
__C.DATA_LOADER.TRAIN_BATCH_SIZE = 32
__C.DATA_LOADER.VALID_BATCH_SIZE = 32
__C.DATA_LOADER.TEST_BATCH_SIZE = 16
__C.DATA_LOADER.NUM_WORKERS = 4
__C.DATA_LOADER.SHUFFLE = True
__C.DATA_LOADER.DROP_LAST = True
__C.DATA_LOADER.PREFETCH_FACTOR = 2
__C.DATA_LOADER.PIN_MEMORY = True
__C.DATA_LOADER.NON_BLOCKING = True

__C.DATA_LOADER.SECOND_STAGE = CfgNode()
__C.DATA_LOADER.SECOND_STAGE.METHOD = ""
__C.DATA_LOADER.SECOND_STAGE.TYPE = ""

# name of dataset type 
__C.DATA_LOADER.DATASET_NAME = ""
__C.DATA_LOADER.NUM_FOLD = 5 # number of fold for cross validation

# ------------------------------------------ step 3 : model config node ------------------------------------------------
__C.MODEL = CfgNode()
__C.MODEL.NAME = "se_resnet50"
__C.MODEL.PRETRAIN = False
__C.MODEL.NUM_CLASS = 2
__C.MODEL.SPECTRUM_CHANNEL = 50
__C.MODEL.RETURN_FEATURE = False
__C.MODEL.AU_CLASS = [12,]
__C.MODEL.BACKBONE_INPUT = 'frame'
__C.MODEL.PREDICTION_FEAT = 'cl'
__C.MODEL.FUSION_TYPE = 'feature_fusion'

# ------------------------------------------ step 4 : loss config node -------------------------------------------------
__C.LOSS = CfgNode()
__C.LOSS.NAME = "mean_square_error"

# ------------------------------------------ step 5 : solver config node -----------------------------------------------
__C.SOLVER = CfgNode()
__C.SOLVER.NAME = "sgd"
__C.SOLVER.RESET_LR = False
__C.SOLVER.LR_INIT = 0.01
__C.SOLVER.WEIGHT_DECAY = 0.0005
__C.SOLVER.MOMENTUM = 0.9
__C.SOLVER.BETA_1 = 0.5
__C.SOLVER.BETA_2 = 0.999
__C.SOLVER.SCHEDULER = "multi_step_scale"
__C.SOLVER.FACTOR = 0.1
__C.SOLVER.MILESTONE = [200, 280]
__C.SOLVER.MODE = 'max'
__C.SOLVER.LR_PATIENCE = 2

# ------------------------------------------- step 6:  train config node -----------------------------------------------
__C.TRAIN = CfgNode()
__C.TRAIN.TRAINER = "ImageModalTrainer"
__C.TRAIN.START_EPOCH = 0
__C.TRAIN.MAX_EPOCH = 300
# __C.TRAIN.PRE_TRAINED_MODEL = None
__C.TRAIN.PRE_TRAINED_MODEL = ""
__C.TRAIN.RESUME = ""
__C.TRAIN.LOG_INTERVAL = 10
__C.TRAIN.VALID_INTERVAL = 1
__C.TRAIN.TEST_INTERVAL = 1
__C.TRAIN.OUTPUT_DIR = "results"
__C.TRAIN.BIMODAL_OPTION = 1 # 1: image only, 2: audio only, 3: image + audio
__C.TRAIN.USE_WANDB = True # True: use wandb, False: not use wandb
__C.TRAIN.TASK = "pretrain_mpc"
__C.TRAIN.ACC_THRESHOLD = 0.8 # accuracy threshold, if best_val_accuracy >= threshold, then save model
__C.TRAIN.SAVED_MODEL_DIR = ""

__C.TRAIN.START_FOLD = 0 # start fold for cross validation
__C.TRAIN.CROSS_RESUME = "" # resume cross validation

__C.TRAIN.USE_AMP = False # use automatic mixed precision
__C.TRAIN.USE_HALF = False # use half precision

# ------------------------------------------- step 7:  test config node ------------------------------------------------
__C.TEST = CfgNode()
__C.TEST.TEST_ONLY = False
__C.TEST.FULL_TEST = False
__C.TEST.WEIGHT = ""
__C.TEST.COMPUTE_PCC = True
__C.TEST.COMPUTE_CCC = True
__C.TEST.SAVE_DATASET_OUTPUT = ""
# ======================================================================================================================


def get_output_dir(imdb, weights_filename):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def get_output_tb_dir(imdb, weights_filename):
    """Return the directory where tensorflow summaries are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """ Merge config dictionary a into config dictionary b, clobbering the
        options in b whenever they are also specified in a.
    """
    if type(a) is not CfgNode:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(
                    'Type mismatch ({} vs. {}) ''for config key: {}'.format(type(b[k]), type(v), k)
                )
        # recursively merge dicts
        if type(v) is CfgNode:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = CfgNode(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
