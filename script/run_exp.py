#! /usr/bin/env python
import sys
import os
import time

current_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.join(current_path, "../")
sys.path.append(work_path)

from dpcv.tools.common import parse_args, setup_seed
from dpcv.config.default_config_opt import cfg, cfg_from_file, cfg_from_list
# from torch.utils.tensorboard import SummaryWriter
from dpcv.experiment.exp_runner import ExpRunner

import wandb

def setup():
    args = parse_args()
    if args.use_wandb == 'True':
        wandb.init(config=args, project="DeepPersonality", settings=wandb.Settings(start_method="fork"))
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.resume:
        cfg.TRAIN.RESUME = args.resume
    if args.max_epoch:
        cfg.TRAIN.MAX_EPOCH = args.max_epoch
    if args.lr:
        cfg.SOLVER.RESET_LR = True
        cfg.SOLVER.LR_INIT = args.lr
    if args.test_only:
        cfg.TEST.TEST_ONLY = True
    if args.bs:
        cfg.DATA_LOADER.TRAIN_BATCH_SIZE = int(args.bs)
    if args.sample_size:
        cfg.DATA.SAMPLE_SIZE = args.sample_size
    if args.use_wandb:
        cfg.TRAIN.USE_WANDB = (args.use_wandb == 'True')
        # print('type(cfg.TRAIN.USE_WANDB): ', type(cfg.TRAIN.USE_WANDB))
    if args.num_workers:
        cfg.DATA_LOADER.NUM_WORKERS = args.num_workers
    if args.prefetch_factor:
        cfg.DATA_LOADER.PREFETCH_FACTOR = args.prefetch_factor
    if args.num_fold:
        cfg.DATA_LOADER.NUM_FOLD = args.num_fold
    if args.au_class:
        # 如果args.au_class是"12,8"这样的字符串，需要转换成[12, 8]这样的列表; 如果args.au_class是"12"这样的字符串，需要转换成[12]这样的列表; 
        if ',' in args.au_class:
            cfg.MODEL.AU_CLASS = [int(i) for i in args.au_class.split(',')]
        else:
            cfg.MODEL.AU_CLASS = [int(args.au_class)]
        print('args.au_class:', args.au_class, 'type(args.au_class):', type(args.au_class), ', cfg.MODEL.AU_CLASS:', cfg.MODEL.AU_CLASS, 'type(cfg.MODEL.AU_CLASS):', type(cfg.MODEL.AU_CLASS))
        # cfg.MODEL.AU_CLASS = args.au_class
    if args.backbone_input:
        cfg.MODEL.BACKBONE_INPUT = args.backbone_input
    if args.prediction_feat:
        cfg.MODEL.PREDICTION_FEAT = args.prediction_feat
    if args.fusion_type:
        cfg.MODEL.FUSION_TYPE = args.fusion_type
    if args.use_amp:
        cfg.TRAIN.USE_AMP = (args.use_amp == 'True')
    if args.use_half:
        cfg.TRAIN.USE_HALF = (args.use_half == 'True')
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    return args


def main():
    setup_seed(12345)
    args = setup()
    print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
    # print('cfg.TRAIN.USE_WANDB:', cfg.TRAIN.USE_WANDB, ', type(cfg.TRAIN.USE_WANDB): ', type(cfg.TRAIN.USE_WANDB))
    if cfg.TRAIN.USE_WANDB == True:
        wandb.config.epoch = cfg.TRAIN.MAX_EPOCH
        wandb.config.frame_per_sec = 5 # 每秒抽取5帧
        wandb.config.cfg = cfg
    # print('args: ', args)
    # print('cfg: ', cfg)
    
    # avoid CUDA out of memory
    if (cfg.MODEL.NAME == "timesformer_udiva" and cfg.DATA_LOADER.TRAIN_BATCH_SIZE >= 16 and cfg.DATA.SAMPLE_SIZE > 16) or cfg.MODEL.NAME in ["visual_graph_representation_learning", "audio_graph_representation_learning", "audiovisual_graph_representation_learning"]:
        print('set os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:{}"'.format(args.max_split_size_mb))
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:{}" .format(args.max_split_size_mb)
    print('os.environ["PYTORCH_CUDA_ALLOC_CONF"] is:', os.environ["PYTORCH_CUDA_ALLOC_CONF"])
    runner = ExpRunner(cfg)
    # print('runner: ', runner)
    if args.test_only:
        return runner.test()
    runner.run()

    ''' print结果
	args:  Namespace(bs=None, cfg_file='./config/demo/bimodal_resnet18.yaml', lr=None, max_epoch=None, resume=None, set_cfgs=None, test_only=False, weight=None)
	cfg:  
	{
		'DATA': {
			'ROOT': 'datasets',
			'TYPE': 'frame',
			'SESSION': 'talk',
			'TRAIN_IMG_DATA': 'ChaLearn2016_tiny/train_data',
			'TRAIN_IMG_FACE_DATA': 'image_data/train_data_face',
			'TRAIN_AUD_DATA': 'ChaLearn2016_tiny/voice_data/voice_librosa/train_data',
			'TRAIN_LABEL_DATA': 'ChaLearn2016_tiny/annotation/annotation_training.pkl',
			'VALID_IMG_DATA': 'ChaLearn2016_tiny/valid_data',
			'VALID_IMG_FACE_DATA': 'image_data/valid_data_face',
			'VALID_AUD_DATA': 'ChaLearn2016_tiny/voice_data/voice_librosa/valid_data',
			'VALID_LABEL_DATA': 'ChaLearn2016_tiny/annotation/annotation_validation.pkl',
			'TEST_IMG_DATA': 'ChaLearn2016_tiny/test_data',
			'TEST_IMG_FACE_DATA': 'image_data/test_data_face',
			'TEST_AUD_DATA': 'ChaLearn2016_tiny/voice_data/voice_librosa/test_data',
			'TEST_LABEL_DATA': 'ChaLearn2016_tiny/annotation/annotation_test.pkl',
			'VA_ROOT': 'datasets',
			'VA_DATA': 'va_data/cropped_aligned',
			'VA_TRAIN_LABEL': 'va_data/va_label/VA_Set/Train_Set',
			'VA_VALID_LABEL': 'va_data/va_label/VA_Set/Validation_Set'
		},
		'DATA_LOADER': {
			'NAME': 'bimodal_resnet_data_loader',
			'DATASET': '',
			'TRANSFORM': 'standard_frame_transform',
			'TRAIN_BATCH_SIZE': 8,
			'VALID_BATCH_SIZE': 4,
			'NUM_WORKERS': 0,
			'SHUFFLE': True,
			'DROP_LAST': True,
			'SECOND_STAGE': {
				'METHOD': '',
				'TYPE': ''
			}
		},
		'MODEL': {
			'NAME': 'audiovisual_resnet',
			'PRETRAIN': False,
			'NUM_CLASS': 5,
			'SPECTRUM_CHANNEL': 50,
			'RETURN_FEATURE': False
		},
		'LOSS': {
			'NAME': 'mean_square_error'
		},
		'SOLVER': {
			'NAME': 'sgd',
			'RESET_LR': False,
			'LR_INIT': 0.001,
			'WEIGHT_DECAY': 0.0005,
			'MOMENTUM': 0.9,
			'BETA_1': 0.5,
			'BETA_2': 0.999,
			'SCHEDULER': 'multi_step_scale',
			'FACTOR': 0.1,
			'MILESTONE': [100, 200]
		},
		'TRAIN': {
			'TRAINER': 'BiModalTrainer',
			'START_EPOCH': 0,
			'MAX_EPOCH': 30,
			'PRE_TRAINED_MODEL': None,
			'RESUME': '',
			'LOG_INTERVAL': 10,
			'VALID_INTERVAL': 1,
			'OUTPUT_DIR': 'results/demo/unified_frame_images/03_bimodal_resnet'
		},
		'TEST': {
			'TEST_ONLY': False,
			'FULL_TEST': False,
			'WEIGHT': '',
			'COMPUTE_PCC': True,
			'COMPUTE_CCC': True,
			'SAVE_DATASET_OUTPUT': ''
		}
	}
	runner:  <dpcv.experiment.exp_runner.ExpRunner object at 0x7f9e68d7a370>
	'''

if __name__ == "__main__":
    # for debug setting
    # import os
    # os.chdir("..")
    main()
