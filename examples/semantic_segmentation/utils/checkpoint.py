"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import torch

from examples.common import restricted_pickle_module
from examples.common.model_loader import NNCF_CHECKPOINT_ATTR
from nncf.torch.checkpoint_loading import load_state


def save_checkpoint(compression_ctrl, optimizer, epoch, miou, compression_stage, config):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - compression_ctrl (``PTCompressionAlgorithmController``): The controller containing compression state to save.
    - optimizer (``torch.optim``): The optimizer state to save.
    - epoch (``int``): The current epoch for the model.
    - miou (``float``): The mean IoU obtained by the model.
    - compression_stage (``CompressionStage``): level of compression
    - compression_scheduler: The compression scheduler associated with the model
    - config: Model config".

    Returns:
        The path to the saved checkpoint.
    """
    name = config.name
    save_dir = config.checkpoint_save_dir

    assert os.path.isdir(
        save_dir), "The directory \"{0}\" doesn't exist.".format(save_dir)

    # Save model
    checkpoint_path = os.path.join(save_dir, name) + "_last.pth"

    checkpoint = {
        'epoch': epoch,
        'miou': miou,
        'compression_stage': compression_stage,
        NNCF_CHECKPOINT_ATTR: compression_ctrl.get_compression_state(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path
