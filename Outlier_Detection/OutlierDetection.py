import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

f_name = '../Real estate valuation data set.xlsx'
# Set K value
k = 5
threshold = 3
iteration = 100
data = pd.read_excel(f_name)

data = data.drop(columns='No')

# Elbow method to determine the K value
'''
K_col = [ x+1 for x in range(15)]
df = pd.read_excel(f_name)
df = df.drop(columns='No')
sse = []
for k in K_col:
    model = KMeans(n_clusters=k).fit(df)
    sse.append(model.inertia_)
sse = np.array(sse)
plt.plot(K_col, sse, marker='o')
plt.xlabel('K')
plt.ylabel('SSE')
plt.title("Elbow Plot")
plt.savefig("../Figures/Elbow Plot.png")
'''
model = KMeans(n_clusters=k, max_iter=iteration)
model.fit(data)

r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
r.columns = list(data.columns) + ["Cluster"]

norm = []
for i in range(k):
    norm_tmp = r[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station',
                     'X4 number of convenience stores', 'X5 latitude', 'X6 longitude',
                     'Y house price of unit area']][r['Cluster'] == i] - model.cluster_centers_[i]
    norm_tmp = norm_tmp.apply(np.linalg.norm, axis=1)
    norm.append(norm_tmp / norm_tmp.median())

norm = pd.concat(norm)

plt.figure(figsize=(12, 8))
plt.title('Outlier Detection')
norm[norm <= threshold].plot(style='bo', legend=True, label='Normal Data')
# df_norm = pd.DataFrame(norm[norm <= threshold], columns='Normal Data')
# df_norm.plot(style='bo')
discrete_points = norm[norm > threshold]
# len(discrete_points) = 56
discrete_points.plot(style='ro', legend='True', label='Outliers')
# plt.legend(handles=[l1,l2],labels=['up','down'],loc='best')
plt.xlabel('No')
plt.ylabel('Relative distance')
plt.tight_layout()
# plt.savefig("../Figures/Outlier Detection.png")
plt.show()
