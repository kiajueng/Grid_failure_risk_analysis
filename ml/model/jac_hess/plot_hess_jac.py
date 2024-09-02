import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

features = ["io_intensity","wall_time","diskio","memory_leak","IObytesWriteRate", "IObytesReadRate","IObytesRead","IObytesWritten","actualcorecount","inputfilebytes","cpu_eff"]
jac = np.load("jacobian.npy")
hess = np.load("hessian.npy")

hess = pd.DataFrame(index=features, columns=features, data = hess)
jac = pd.DataFrame(data=jac, columns = features)

# Define the plot
fig, ax = plt.subplots(figsize=(25,10))

sns.heatmap(hess, annot=True, cmap="viridis", ax=ax, fmt=".1f")
plt.savefig("Hessian.pdf", bbox_inches="tight")

plt.close(fig)

# Define the plot
fig, ax = plt.subplots(figsize=(25,5))
ax.set_yticks([])
sns.heatmap(jac, annot=True, cmap="viridis", ax=ax, fmt=".1f")

plt.savefig("jacobian.pdf", bbox_inches="tight")

plt.close(fig)
