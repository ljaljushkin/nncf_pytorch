# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum

from nncf.common.utils.api_marker import api


@api(canonical_alias="nncf.TargetDevice")
class TargetDevice(Enum):
    """
    Target device architecture for compression.

    Compression will take into account the value of this parameter in order to obtain the best performance
    for this type of device.
    """

    ANY = "ANY"
    CPU = "CPU"
    GPU = "GPU"
    VPU = "VPU"
    CPU_SPR = "CPU_SPR"


@api(canonical_alias="nncf.ModelType")
class ModelType(Enum):
    """
    Describes the model type the specificity of which will be taken into account during compression.

    :param TRANSFORMER: Transformer-based models
        (https://arxiv.org/pdf/1706.03762.pdf)
    """

    TRANSFORMER = "transformer"


@api(canonical_alias="nncf.DropType")
class DropType(Enum):
    """
    Describes the accuracy drop type, which determines how the accuracy drop between
    the original model and the compressed model is calculated.

    :param ABSOLUTE: The accuracy drop is calculated as the absolute drop with respect
        to the results of the original model.
    :param RELATIVE: The accuracy drop is calculated relative to the results of
        the original model.
    """

    ABSOLUTE = "absolute"
    RELATIVE = "relative"


@api(canonical_alias="nncf.CompressWeightsMode")
class CompressWeightsMode(Enum):
    """
    Defines a mode for weight compression.

    :param INT8: Stands for 8-bit integer quantization of all weights.
    :param INT4: Stands for 4-bit integer quantization of all weights.
    :param NF4: Stands for a mixed-precision weights quantization to NF4 data type. The first and last
        layers are always compressed to a backup precision which is 8-bit integer by default. All others are quantized
        whether to NF4 or to a backup precision depending on criteria and the given ratio.
    """

    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"
