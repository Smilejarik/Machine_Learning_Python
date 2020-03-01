import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500  # dog type
labs = 500  # dog type

grey_height = 28 + 4*np.random.randn(greyhounds)  # average height plus random
lab_height = 24 + 4*np.random.randn(labs)  # average height plus random

# vis plot will show us overlapping in the middle with about 50/50 probability
vis_out = plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show(vis_out)