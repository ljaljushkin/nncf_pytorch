import torch.backends.cudnn as cudnn

from examples.common.model_loader import load_checkpoints_from_path
from nncf.utils import manual_seed


def set_seed(config):
    if config.seed is not None:
        manual_seed(config.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def load_resuming_checkpoint(resuming_checkpoint_path):
    nncf_checkpoint = None
    resuming_checkpoint = None
    if resuming_checkpoint_path is not None:
        nncf_checkpoint, resuming_checkpoint = load_checkpoints_from_path(
            resuming_checkpoint_path)
    return nncf_checkpoint, resuming_checkpoint
