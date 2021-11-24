import os
from datetime import datetime
from dpcv.data.datasets.build import build_dataloader
from dpcv.modeling.networks.build import build_model
from dpcv.modeling.loss.build import build_loss_func
from dpcv.modeling.solver.build import build_solver, build_scheduler
from dpcv.engine.build import build_trainer
from dpcv.config.default_config_opt import cfg as test_cfg
from dpcv.evaluation.summary import TrainSummary
from dpcv.checkpoint.save import save_model, resume_training, load_model
from dpcv.evaluation.metrics import compute_pcc, compute_ccc
from dpcv.tools.logger import make_logger


class ExpRunner:

    def __init__(self, cfg):
        """ run exp from config file

        arg:
            cfg_file: config file of an experiment
        """

        """
        construct certain experiment by the following template
        step 1: prepare dataloader
        step 2: prepare model and loss function
        step 3: select optimizer for gradient descent algorithm
        step 4: prepare trainer for typical training in pytorch manner
        """
        self.cfg = cfg
        self.logger, self.log_dir = make_logger(cfg.TRAIN.OUTPUT_DIR)

        self.data_loader = self.build_dataloader()

        self.model = self.build_model()
        self.loss_f = self.build_loss_function()

        self.optimizer = self.build_solver()
        self.scheduler = self.build_scheduler()

        self.collector = TrainSummary()
        self.trainer = self.build_trainer()

    def build_dataloader(self):
        dataloader = build_dataloader(self.cfg)
        data_loader_dicts = {
            "train": dataloader(self.cfg, mode="train"),
            "valid": dataloader(self.cfg, mode="valid"),
            "test": dataloader(self.cfg, mode="test"),
        }
        return data_loader_dicts

    def build_model(self):
        return build_model(self.cfg)

    def build_loss_function(self):
        return build_loss_func(self.cfg)

    def build_solver(self):
        return build_solver(self.cfg, self.model)

    def build_scheduler(self):
        return build_scheduler(self.cfg, self.optimizer)

    def build_trainer(self):
        return build_trainer(self.cfg, self.collector, self.logger)

    def train(self):
        cfg = self.cfg.TRAIN
        if cfg.RESUME:
            self.model, self.optimizer, epoch = resume_training(cfg.RESUME, self.model, self.optimizer)
            cfg.START_EPOCH = epoch
            self.logger.info(f"resume training from {cfg.RESUME}")

        for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
            self.trainer.train(self.data_loader["train"], self.model, self.loss_f, self.optimizer, epoch)
            self.trainer.valid(self.data_loader["valid"], self.model, self.loss_f, epoch)
            self.scheduler.step()

            if self.collector.model_save:
                save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.log_dir, cfg)
                self.collector.update_best_epoch(epoch)

        self.collector.draw_epo_info(cfg.MAX_EPOCH - cfg.START_EPOCH, self.log_dir)
        self.logger.info(
            "{} done, best acc: {} in :{}".format(
                datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                self.collector.best_valid_acc,
                self.collector.best_epoch,
            )
        )

    def test(self, weight=None):
        self.logger.info("Test only mode")
        cfg = self.cfg.TEST
        cfg.WEIGHT =  weight if weight else cfg.WEIGHT

        if cfg.WEIGHT:
            self.model = load_model(self.model, cfg.WEIGHT)
        else:
            weights = sorted(
                [file for file in os.listdir(self.log_dir) if file.endswith(".pkl") and ("last" not in file)]
            )
            weight_file = os.path.join(self.log_dir, weights[-1])
            self.model = load_model(self.model, weight_file)

        ocean_acc_avg, ocean_acc, dataset_output, dataset_label = self.trainer.test(
            self.data_loader["test"], self.model
        )
        self.logger.info("acc: {} mean: {}".format(ocean_acc, ocean_acc_avg))

        if cfg.COMPUTE_PCC:
            pcc_dict, pcc_mean = compute_pcc(dataset_output, dataset_label)
            self.logger.info(f"pcc: {pcc_dict} mean: {pcc_mean}")

        if cfg.COMPUTE_CCC:
            ccc_dict, ccc_mean = compute_ccc(dataset_output, dataset_label)
            self.logger.info(f"ccc: {ccc_dict} mean: {ccc_mean}")
        return

    def run(self):
        self.train()
        self.test()


if __name__ == "__main__":
    # args = parse_args()
    import os
    import torch
    os.chdir("/home/rongfan/05-personality_traits/DeepPersonality")

    exp_runner = ExpRunner(test_cfg)
    xin = torch.randn((1, 3, 224, 224)).cuda()
    y = exp_runner.model(xin)
    print(y.shape)
    # main(args, cfg)
