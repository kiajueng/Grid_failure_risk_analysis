import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#features = ["io_intensity","wall_time","diskio","memory_leak","IObytesWriteRate", "IObytesReadRate","IObytesRead","IObytesWritten","actualcorecount","inputfilebytes","cpu_eff"]
name = "prediction_weights_abs"
features = []

with open(f"features_{name}.txt","r+") as f:
    feat = f.read().split("\n")
    features = [l for l in feat if l != ""]

jac = np.load(f"jacobian_{name}.npy")
hess = np.load(f"hessian_{name}.npy")

hess = pd.DataFrame(index=features, columns=features, data = hess)
jac = pd.DataFrame(data=jac, columns = features)

# Define the plot
fig, ax = plt.subplots(figsize=(25,10))
plt.xticks(fontsize = 13) 
plt.yticks(fontsize = 13) 
sns.set(font_scale=1.3)
sns.heatmap(hess, annot=True, cmap="viridis", ax=ax, fmt=".1f")

plt.savefig(f"Hessian_{name}.pdf", bbox_inches="tight")

plt.close(fig)

# Define the plot
fig, ax = plt.subplots(figsize=(25,5))
ax.set_yticks([])
sns.heatmap(jac, annot=True, cmap="viridis", ax=ax, fmt=".1f")

plt.savefig(f"jacobian_{name}.pdf", bbox_inches="tight")

plt.close(fig)
