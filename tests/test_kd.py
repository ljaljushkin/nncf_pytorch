import numpy as np
import tensorflow as tf
import torch

n1 = np.random.rand(3, 4, 2, 2)
n2 = np.random.rand(3, 4, 2, 2)

t2 = tf.convert_to_tensor(n1)
s2 = tf.convert_to_tensor(n2)
tab2 = tf.nn.softmax(t2, axis=-1)
sab2 = tf.nn.log_softmax(s2, axis=-1)
kl2 = -(tab2 * sab2)
sum2 = tf.reduce_sum(kl2, axis=-1)
m2 = tf.reduce_mean(sum2)

t1 = torch.from_numpy(n1)
s1 = torch.from_numpy(n2)
tab1 = torch.nn.functional.softmax(t1, dim=-1)
sab1 = torch.nn.functional.log_softmax(s1, dim=-1)
kl1 = -(tab1 * sab1)
# sum1 = kl1.sum(dim=-1)
m1 = kl1.mean() * s1.shape[-1]
#  -(nn.functional.log_softmax(student_output, dim=1) * nn.functional.softmax(teacher_output / temperature, dim=1)
#                     ).mean()
#                     * student_output.shape[1]

print(f"{m2} vs {m1}")
