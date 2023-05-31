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
"""
Contains builder and controller class definitions for the knowledge distillation.
"""

from copy import deepcopy
from typing import Dict

from torch import nn

from nncf import NNCFConfig, nncf_logger
from nncf.api.compression import CompressionLoss, CompressionScheduler, CompressionStage
from nncf.common.graph.transformations.commands import TargetType, TransformationPriority
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.scopes import should_consider_scope
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.api_marker import api
from nncf.config.schemata.defaults import KNOWLEDGE_DISTILLATION_SCALE, KNOWLEDGE_DISTILLATION_TEMPERATURE
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder, PTCompressionAlgorithmController
from nncf.torch.dynamic_graph.patch_pytorch import register_operator
from nncf.torch.graph.transformations.commands import PTInsertionCommand, PTTargetPoint
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.knowledge_distillation.knowledge_distillation_loss import KnowledgeDistillationLoss
from nncf.torch.nncf_network import NNCFNetwork, PTModelTransformer


@register_operator()
def collect_dummy(output):
    return output


class OutputCollector(nn.Module):
    def __init__(self):
        super().__init__()
        # self.dummy_param = Parameter(torch.zeros(1))
        self._output_storage = None

    @property
    def output(self):
        return self._output_storage

    def forward(self, output):
        self._output_storage = output
        return collect_dummy(output)


@PT_COMPRESSION_ALGORITHMS.register("knowledge_distillation")
class KnowledgeDistillationBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)
        self.kd_type = self._algo_config.get("type")
        self.scale = self._algo_config.get("scale", KNOWLEDGE_DISTILLATION_SCALE)
        self.temperature = self._algo_config.get("temperature", KNOWLEDGE_DISTILLATION_TEMPERATURE)
        self.a_scopes = self._algo_config.get("a_scopes")
        self.h_scopes = self._algo_config.get("h_scopes")
        # if "temperature" in self._algo_config.keys() and self.kd_type == "mse":
        #     raise ValueError("Temperature shouldn't be stated for MSE Loss (softmax only feature)")
        if self.kd_type == "none":
            self.kd_type = None
            nncf_logger.info("NL: No distillation by output")
        self.a_student_collectors = {}
        self.a_teacher_collectors = {}
        self.h_student_collectors = {}
        self.h_teacher_collectors = {}

    def _create_layout(self, graph, scopes, layout):
        collectors = {}
        nncf_logger.info(f"KD for intermediate layers:")
        for node in graph.get_all_nodes():
            node_name = node.node_name
            if should_consider_scope(node_name, None, scopes):
                nncf_logger.info(f"{node_name}")
                op = OutputCollector()
                collectors[node_name] = op
                command = PTInsertionCommand(
                    PTTargetPoint(TargetType.OPERATOR_POST_HOOK, target_node_name=node_name),
                    op,
                    TransformationPriority.QUANTIZATION_PRIORITY,
                )
                layout.register(command)
        return collectors

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        graph = target_model.nncf.get_original_graph()
        self.original_model = deepcopy(target_model).nncf.get_clean_shallow_copy()
        student_layout = PTTransformationLayout()
        if self.a_scopes or self.h_scopes:
            teacher_layout = PTTransformationLayout()
            if self.a_scopes:
                self.a_student_collectors = self._create_layout(graph, self.a_scopes, student_layout)
                self.a_teacher_collectors = self._create_layout(graph, self.a_scopes, teacher_layout)
            if self.h_scopes:
                self.h_student_collectors = self._create_layout(graph, self.h_scopes, student_layout)
                self.h_teacher_collectors = self._create_layout(graph, self.h_scopes, teacher_layout)

            transformer = PTModelTransformer(self.original_model)
            self.original_model = transformer.transform(teacher_layout)

        for param in self.original_model.parameters():
            param.requires_grad = False
        return student_layout

    def _build_controller(self, model):
        return KnowledgeDistillationController(
            model,
            self.original_model,
            self.kd_type,
            self.scale,
            self.temperature,
            self.a_student_collectors,
            self.a_teacher_collectors,
            self.h_student_collectors,
            self.h_teacher_collectors,
        )

    def initialize(self, model: NNCFNetwork) -> None:
        pass


@api()
class KnowledgeDistillationController(PTCompressionAlgorithmController):
    """
    Controller for the knowledge distillation in PT.
    """

    def __init__(
        self,
        target_model: NNCFNetwork,
        original_model: nn.Module,
        kd_type: str,
        scale: float,
        temperature: float,
        a_student_collectors: Dict[str, OutputCollector],
        a_teacher_collectors: Dict[str, OutputCollector],
        h_student_collectors: Dict[str, OutputCollector],
        h_teacher_collectors: Dict[str, OutputCollector],
    ):
        super().__init__(target_model)
        original_model.train()
        self._scheduler = BaseCompressionScheduler()
        self._loss = KnowledgeDistillationLoss(
            target_model=target_model,
            original_model=original_model,
            kd_type=kd_type,
            scale=scale,
            temperature=temperature,
            a_student_collectors=a_student_collectors,
            a_teacher_collectors=a_teacher_collectors,
            h_student_collectors=h_student_collectors,
            h_teacher_collectors=h_teacher_collectors,
        )

    def compression_stage(self) -> CompressionStage:
        """
        Returns level of compression. Should be used on saving best checkpoints to distinguish between
        uncompressed, partially compressed and fully compressed models.
        """
        return CompressionStage.FULLY_COMPRESSED

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()

    def distributed(self):
        pass
