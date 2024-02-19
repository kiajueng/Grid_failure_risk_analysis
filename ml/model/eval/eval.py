import pandas as pd
import seaborn as sns
import glob
import matplotlib.pyplot as plt

data = []

for f in glob.glob("/share/scratch1/kiajueng_yang/data_pred/*2023_11*"):
    data.append(pd.read_csv(f,usecols=["cpu_eff","jobstatus","prediction","new_weights"]))

data = pd.concat(data)
data.loc[:,"jobstatus"] = data.loc[:,"jobstatus"].map({0.:"failed",1.:"finished"})
data = data[data.cpu_eff > 0.05]
data = data[(data.jobstatus=="finished") | ((data.new_weights > 10/3 * 0.5) & (data.jobstatus == "failed"))]

f,ax = plt.subplots(figsize=(10,5))
sns.histplot(data=data,x="prediction",hue="jobstatus",bins=20,alpha=0,element="step",stat="probability", ax=ax, common_norm = False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1.04, 1))
plt.savefig("pred.pdf",bbox_inches="tight")
