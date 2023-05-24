import numpy as np
import tensorflow as tf
import torch

n = np.random.rand(3, 4, 2)

t2 = tf.convert_to_tensor(n)
ln2 = tf.keras.layers.LayerNormalization(t2, axis=-1)
# contrib_layers.layer_norm(
#                       inputs=teacher_hidden,
#                       begin_norm_axis=-1,
#                       begin_params_axis=-1,
#                       trainable=False)),


t1 = torch.from_numpy(n)
