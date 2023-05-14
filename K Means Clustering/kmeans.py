import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# df = pd.DataFrame({
#     'x' : [12, 20, 28, 18, 29, 33, 14, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
#     'y' : [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
# })

# np.random.seed(200)
# k=3
# #centroids[i] = [x, y]
# centroids = {
#     i+1: [np.random.randint(0, 80), np.random.randint(0, 80)]
#     for i in range(k)
# }

# fig = plt.figure(figsize=(5, 5))
# plt.scatter(df['x'], df['y'], color='k')
# colmap = {1: 'r', 2: 'g', 3: 'b'}
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()

# # Assignment Stage

# def assignment(df, centroids):
#     for i in centroids.keys():
#         #sqrt((x1 - x2)^2 - (y1 - y2)^2)
#         df['distance_from_{}'.format(i)] = (
#             np.sqrt(
#                 (df['x'] - centroids[i][0]) ** 2
#                 + (df['y'] - centroids[i][1]) ** 2
#             )
#         )
#     centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
#     df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
#     df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
#     df['color'] = df['closest'].map(lambda x: colmap[x])
#     return df

# df = assignment(df, centroids)
# print(df.head())

# fig = plt.figure(figsize=(5, 5))
# plt.scatter(df['x'], df['f'], color=df['color'], alpha=0.5, edgecolor='k')
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()

# #Repeat Assignment Stage

# df = assignment(df, centroids)

# #plot results
# fig = plt.figure(figsize=(5, 5))
# plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()

# #continue until all assigned categories don't change any more
# while True:
#     closest_centroids = df['closest'].copy(deep=True)
#     centroids = update(centroids)
#     df = assignment(df, centroids)
#     if closest_centroids.equals(df['closest']):
#         break
    
# fig = plt.figure(figsize=(5, 5))
# plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()

df = pd.DataFrame({
    'x' : [12, 20, 28, 18, 29, 33, 14, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y' : [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_
colmap = {1: 'r', 2: 'g', 3: 'b'}

fig = plt.figure(figsize=(5, 5))

colors = map(lambda x: colmap[x+1], labels)
colors1 = list(colors)
plt.scatter(df['x'], df['y'], color=colors1, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()
