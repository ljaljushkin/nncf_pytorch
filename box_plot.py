from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

save_dir = Path('/home/nlyaly/projects/lm-evaluation-harness/saved_activations')
filename = '__module.model.lm_head.aten::linear.MatMul_1957.npy'
all_data = np.load(save_dir / filename)
all_data.shape
# generate some random test data
all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]

fig, ax = plt.subplots(figsize=(4, 4))
# plot box plot
ax.boxplot(all_data)
ax.set_title('Box plot')

# ax.yaxis.grid(True)
ax.set_xticks([y + 1 for y in range(len(all_data))],
              labels=['x1', 'x2', 'x3', 'x4'])
ax.set_xlabel('Four separate samples')
ax.set_ylabel('Observed values')

plt.show()