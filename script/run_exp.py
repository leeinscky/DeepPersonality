#! /usr/bin/env python
import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.join(current_path, "../")
sys.path.append(work_path)

from dpcv.tools.common import parse_args
from dpcv.config.default_config_opt import cfg, cfg_from_file, cfg_from_list
# from torch.utils.tensorboard import SummaryWriter
from dpcv.experiment.exp_runner import ExpRunner

import wandb

def setup():
    args = parse_args()
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

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    return args


def main():
    args = setup()
    # print('args: ', args)
    # print('cfg: ', cfg)
    # print('[DeepPersonality/script/run_exp.py] - 开始执行 runner = ExpRunner(cfg)')
    runner = ExpRunner(cfg)
    # print('[DeepPersonality/script/run_exp.py] - 结束执行 runner = ExpRunner(cfg)')
    # print('[DeepPersonality/script/run_exp.py] - 开始执行 runner.run()')
    # print('runner: ', runner)
    if args.test_only:
        return runner.test()
    runner.run()
    # print('[DeepPersonality/script/run_exp.py] - 结束执行 runner.run()')

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
