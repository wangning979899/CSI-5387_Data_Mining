import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN

f_name = '../Real estate valuation data set.xlsx'
# Set K value
k_value = 8
threshold = 2
iteration = 100
data = pd.read_excel(f_name)

data = data.drop(columns='No')

# Elbow method to determine the K value
K_col = [ x+1 for x in range(15)]
# df = pd.read_excel(f_name)
# df = df.drop(columns='No')
df = MinMaxScaler().fit_transform(data)
df = pd.DataFrame(df, columns=data.columns)
sse = []
for k in K_col:
    model = KMeans(n_clusters=k).fit(df)
    sse.append(model.inertia_)
sse = np.array(sse)
plt.plot(K_col, sse, marker='o')
plt.xlabel('K')
plt.ylabel('SSE')
plt.title("Elbow Plot")
# plt.savefig("../Figures/Elbow Plot.png")

# Standardize dataset
data_std = MinMaxScaler().fit_transform(data)
data_std = pd.DataFrame(data_std, columns=data.columns)

model = KMeans(n_clusters=k_value, max_iter=iteration)
# model.fit(data)
model.fit(data_std)

# r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
# r.columns = list(data.columns) + ["Cluster"]
r = pd.concat([data_std, pd.Series(model.labels_, index=data_std.index)], axis=1)
r.columns = list(data_std.columns) + ["Cluster"]
norm = []

for i in range(k_value):
    norm_tmp = r[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station',
                  'X4 number of convenience stores', 'X5 latitude', 'X6 longitude',
                  'Y house price of unit area']][r['Cluster'] == i] - model.cluster_centers_[i]
    # Use L2-Norm to calculate the absolute distance
    norm_tmp = norm_tmp.apply(np.linalg.norm, axis=1)
    norm.append(norm_tmp / norm_tmp.median())

norm = pd.concat(norm)

plt.figure(figsize=(12, 8))
plt.title('Outlier Detection')
norm[norm <= threshold].plot(style='bo', legend=True, label='Normal Data')
discrete_points = norm[norm > threshold]
# len(discrete_points) = 15
discrete_points.plot(style='ro', legend='True', label='Outliers')
# plt.legend(handles=[l1,l2],labels=['up','down'],loc='best')
plt.xlabel('No')
plt.ylabel('Relative distance')
plt.tight_layout()
# plt.savefig("../Figures/Outlier Detection.png")
plt.show()



# DBSCAN
model_d = DBSCAN()
model_d.fit(data_std)
r2 = pd.concat([data_std, pd.Series(model_d.labels_, index=data_std.index)], axis=1)
r2.columns = list(data_std.columns) + ["Cluster"]
# Check the outliers
# The result shows that the outlier is No. 270 whose price is 117.5
for i in range(len(r2)):
    if (r2['Cluster'][i]==-1):
        print(data.iloc[i, ])

# Draw the plot
labels = model_d.labels_
labels = pd.Series(labels)
plt.figure(figsize=(12, 8))
plt.title('Outlier Detection (DBSCAN)')
labels[labels>=0].plot(style='bo', legend=True, label='Normal Data')
labels[labels < 0].plot(style='ro', legend=True, label='Outliers')
plt.xlabel('No')
plt.ylabel('Labels')
plt.show()

# removing the outliers
'''
for i in discrete_points.keys():
    print(data.loc[i])
    
count = []
for i in range(len(r2)):
    if (r2['Cluster'][i]==-1):
        print(data.iloc[i, ])
        count.append(i)
for i in count:
    data.drop(i)
'''
