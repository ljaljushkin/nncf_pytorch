from typing import Any
from typing import Dict
from typing import List

from nncf.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim


class StageDescriptor:
    def __init__(self, train_dims: List[ElasticityDim],
                 epochs: int = 1,
                 reorg_weights: bool = False,
                 bn_adapt: bool = False,
                 depth_indicator: int = 1,
                 width_indicator: int = 1):
        self.train_dims = train_dims
        self.epochs = epochs
        self.depth_indicator = depth_indicator
        self.width_indicator = width_indicator
        self.reorg_weights = reorg_weights
        self.bn_adapt = bn_adapt

    def __eq__(self, other: 'StageDescriptor'):
        return self.train_dims == other.train_dims and \
               self.epochs == other.epochs and \
               self.reorg_weights == other.reorg_weights and \
               self.width_indicator == other.width_indicator and \
               self.depth_indicator == other.depth_indicator and \
               self.bn_adapt == other.bn_adapt

    @classmethod
    def from_state(cls, state: Dict[str, Any]):
        new_dict = state.copy()
        train_dims = state.get('train_dims', {})
        new_dict['train_dims'] = [ElasticityDim.from_str(dim) for dim in train_dims]
        return cls(**new_dict)

    def get_state(self) -> Dict[str, Any]:
        return {
            'train_dims': [dim.value for dim in self.train_dims],
            'epochs': self.epochs,
            'reorg_weights': self.reorg_weights,
            'width_indicator': self.width_indicator,
            'depth_indicator': self.depth_indicator,
            'bn_adapt': self.bn_adapt,
        }