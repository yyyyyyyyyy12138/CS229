import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


labels = np.load("labels.npy")
preds = np.load("preds.npy")

matrix = confusion_matrix(preds, labels)

sns.heatmap(matrix, cmap="", annot=True)
plt.show()

