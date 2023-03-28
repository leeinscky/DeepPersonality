import os
import torch


def save_model(epoch, best_acc, model, optimizer, output_dir, cfg, fold_id=None):
    if isinstance(optimizer, list):
        optimizer = optimizer[1]  # for cr net
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc
    }
    if fold_id is not None:
        pkl_name = "checkpoint_fold{}_acc{}_ep{}.pkl".format(fold_id, best_acc, epoch) if epoch != (cfg.MAX_EPOCH - 1) else "checkpoint_last_fold{}_acc{}.pkl".format(fold_id, best_acc)
    else:
        pkl_name = "checkpoint_acc{}_ep{}.pkl".format(best_acc, epoch) if epoch != (cfg.MAX_EPOCH - 1) else "checkpoint_last_acc{}.pkl".format(best_acc)
    path_checkpoint = os.path.join(output_dir, pkl_name)
    torch.save(checkpoint, path_checkpoint)


def resume_training(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return model, optimizer, epoch


def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


