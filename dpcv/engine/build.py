from dpcv.tools.registry import Registry

print('[DeepPersonality/dpcv/engine/build.py] start')
print('[DeepPersonality/dpcv/engine/build.py] 准备执行：TRAINER_REGISTRY = Registry("TRAINER")')
TRAINER_REGISTRY = Registry("TRAINER")
print('[DeepPersonality/dpcv/engine/build.py] 执行：TRAINER_REGISTRY = Registry("TRAINER") 结束')


def build_trainer(cfg, collector, logger):
    print('[DeepPersonality/dpcv/engine/build.py] - 开始执行 def build_trainer(cfg, collector, logger)')
    name = cfg.TRAIN.TRAINER # 'TRAINER': 'BiModalTrainer',
    trainer_cls = TRAINER_REGISTRY.get(name)
    print('[DeepPersonality/dpcv/engine/build.py] - 结束执行 def build_trainer(cfg, collector, logger)')
    return trainer_cls(cfg.TRAIN, collector, logger)
