import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("customer_product_list.csv", sep=',', encoding='iso8859_9')
data_df = pd.DataFrame(data, columns=['user_id', 'ProductTypeID', 'sum_quantity', 'total_price' ])

X = data.iloc[:,2:].values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, linewidth=1, color="red", marker ="8")
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++',max_iter=300, n_init=10, random_state=0)
pred = kmeans.fit_predict(X)
plt.scatter(X[pred==0,0],X[pred==0,1], s=60, c='red')
plt.scatter(X[pred==1,0],X[pred==1,1], s=60, c='blue')
plt.scatter(X[pred==2,0],X[pred==2,1], s=60, c='green')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=200,marker='s', c='red', alpha=0.7, label='Centroids')
plt.title('KMeans')
plt.show()

frame = pd.DataFrame(X)
frame['cluster'] = pred
print(frame)
print(frame['cluster'].value_counts())
print("INERTIA", kmeans.inertia_)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[pred==0,0],X[pred==0,1], s=60, c='red')
ax.scatter(X[pred==1,0],X[pred==1,1], s=60, c='blue')
ax.scatter(X[pred==2,0],X[pred==2,1], s=60, c='green')
plt.xlabel("Sum quantity")
plt.ylabel("Total Price")
plt.show()