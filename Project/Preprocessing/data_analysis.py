import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_excel("./Real estate valuation data set.xlsx")
dataset = dataset.drop('No', axis=1)
df_describe = dataset.describe()
print(df_describe)

# Histogram
dataset.hist()
plt.tight_layout(rect=(0, 0, 3, 3))
# plt.show()
plt.savefig('./Figures/Historgrams_total1.png', bbox_inches='tight')

dataset['X1 transaction date'].hist()

# Continuous distribution curves
columns = dataset.columns
fig, axs = plt.subplots(3, 3, figsize= (20, 20))
title = fig.suptitle("Continuous distribution",y=0.9, va='center',fontsize=27)

# the flag judge whether the attribute is visitied
count=0
for i in range(3):
    for j in range(3):
        if (count <= 6):
            axs[i, j].set_xlabel(columns[count], fontsize=18)
            axs[i, j].set_ylabel("Frequency", fontsize=18)
            sns.kdeplot(dataset[columns[count]], ax=axs[i, j], shade=True)
            count += 1
        else:
            # Hide axis and boarders of the last 2 subplots.
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['bottom'].set_visible(False)
            axs[i, j].spines['left'].set_visible(False)
            axs[i, j].axis('off')
plt.savefig("./Figures/Continours_distribution.png")
# plt.show()

# Heatmap
fig, ax = plt.subplots(figsize=(10, 6))
corr = dataset.corr()
hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm",
                 fmt='.2f', linewidths=.05)
fig.subplots_adjust()
title = fig.suptitle("Attributes Correlation Heatmap", x=0.3, fontsize=20)
plt.savefig('./Figures/Heatmap.png', bbox_inches='tight', dpi=500)
# plt.show()

# Box plot
# Only categorical attribute can be used as the x-label.
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
title = fig.suptitle("Box-plot", fontsize=25)
sns.boxplot(x='X1 transaction date', y='Y house price of unit area', data=dataset, ax=ax)
plt.savefig('./Figures/Box-plot.png',bbox_inches='tight', dpi=300)
# plt.show()