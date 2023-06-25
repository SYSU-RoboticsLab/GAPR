# import torch.nn.functional as F
# import torch

# # test padding
# t4d = torch.randn(10, 3)
# p1d = (0,0,0,5)
# out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
# print(out.size(),out)

# indices = range(16)
# res = indices[0:16:4]

# print(res)



# a = np.array([10, 9, 11, 12, 5, 1, 3])
# b = np.array([1,2,3,4,5,6,7,8,9])
# print(np.intersect1d(a, b, return_indices=True))
# import numpy as np
# import folium

# from folium.plugins import HeatMap
# data = (
#     np.random.normal(size=(100, 3)) *
#     np.array([[0.1, 0.1, 1]]) +
#     np.array([[40, 116.5, 0]])
# )
# print(data)

# m = folium.Map([39.93, 116.38], zoom_start=6)
# HeatMap(data).add_to(m)
# m.save("misc/Heatmap.html")

# import branca.colormap
# from collections import defaultdict
# import folium
# import webbrowser
# from folium.plugins import HeatMap

# map_osm = folium.Map(llocation=[35,110],zoom_start=1)

# steps=20
# colormap = branca.colormap.linear.YlOrRd_09.scale(0, 1).to_step(steps)
# gradient_map=defaultdict(dict)
# for i in range(steps):
#     gradient_map[1/steps*i] = colormap.rgb_hex_str(1/steps*i)
# colormap.add_to(map_osm) #add color bar at the top of the map

# HeatMap(data1,gradient = gradient_map).add_to(map_osm) # Add heat map to the previously created map

# file_path = r"C:\\\\test.html"
# map_osm.save(file_path) # Save as html file
# webbrowser.open(file_path) # Default browser open


# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np

# plt.figure(figsize=(13, 4))
# # 构造x轴刻度标签、数据
# labels = ['G1', 'G2', 'G3', 'G4', 'G5']
# first = [20, 34, 30, 35, 27]
# second = [25, 32, 34, 20, 25]
# third = [21, 31, 37, 21, 28]
# fourth = [26, 31, 35, 27, 21]

# # 两组数据
# plt.subplot(131)
# x = np.arange(len(labels))  # x轴刻度标签位置
# width = 0.25  # 柱子的宽度
# # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
# # x - width/2，x + width/2即每组数据在x轴上的位置
# plt.bar(x - width/2, first, width, label='1')
# plt.bar(x + width/2, second, width, label='2')
# plt.ylabel('Scores')
# plt.title('2 datasets')
# # x轴刻度标签位置不进行计算
# plt.xticks(x, labels=labels)
# plt.legend()
# # 三组数据
# plt.subplot(132)
# x = np.arange(len(labels))  # x轴刻度标签位置
# width = 0.25  # 柱子的宽度
# # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
# # x - width，x， x + width即每组数据在x轴上的位置
# plt.bar(x - width, first, width, label='1')
# plt.bar(x, second, width, label='2')
# plt.bar(x + width, third, width, label='3')
# plt.ylabel('Scores')
# plt.title('3 datasets')
# # x轴刻度标签位置不进行计算
# plt.xticks(x, labels=labels)
# plt.legend()
# # 四组数据
# plt.subplot(133)
# x = np.arange(len(labels))  # x轴刻度标签位置
# width = 0.2  # 柱子的宽度
# # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
# plt.bar(x - 1.5*width, first, width, label='1')
# plt.bar(x - 0.5*width, second, width, label='2')
# plt.bar(x + 0.5*width, third, width, label='3')
# plt.bar(x + 1.5*width, fourth, width, label='4')
# plt.ylabel('Scores')
# plt.title('4 datasets')
# # x轴刻度标签位置不进行计算
# plt.xticks(x, labels=labels)
# plt.legend()

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV

# def generate_data(seed=17):
#     # Fix the seed to reproduce the results
#     rand = np.random.RandomState(seed)
#     x = []
#     dat = rand.lognormal(0, 0.3, 1000)
#     x = np.concatenate((x, dat))
#     dat = rand.normal(3, 1, 1000)
#     x = np.concatenate((x, dat))
#     return x

# x_train = generate_data()[:, np.newaxis]
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# plt.subplot(121)
# plt.scatter(np.arange(len(x_train)), x_train, c='green')
# plt.xlabel('Sample no.')
# plt.ylabel('Value')
# plt.title('Scatter plot')
# plt.subplot(122)
# plt.hist(x_train, bins=50)
# plt.title('Histogram')
# fig.subplots_adjust(wspace=.3)
# plt.show()

# def funb(a=11, b=12, c=13):
#     print(c, b, a)

# def func(name, **kw1):
#     print(name)
#     print(list(kw1.keys()))
#     return


# if __name__ == "__main__":
#     param = {"a":1, "b":"2", "c":{"ca":1, "cb":2}, "name":"myname"}
#     func(**param)

# a = (1,2,3,4,)

# # for i in a:
# #     print(i)
# import open3d as o3d
# import numpy as np
# import time


# N = 2000


# source_pcd = o3d.t.geometry.PointCloud()
# source_pcd.point["positions"] = o3d.core.Tensor(np.random.rand(N, 3)*3)
# source_pcd = source_pcd.cuda(0)

# target_pcd = o3d.t.geometry.PointCloud()
# target_pcd.point["positions"] = o3d.core.Tensor(np.random.rand(N, 3)*3)
# target_pcd = target_pcd.cuda(0)


# estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
# # Search distance for Nearest Neighbour Search [Hybrid-Search is used].
# max_correspondence_distance = 0.2

# # Initial alignment or source to target transform.
# init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)

# # Convergence-Criteria for Vanilla ICP
# criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0000001, relative_rmse=0.0000001, max_iteration=30)

# # Down-sampling voxel-size. If voxel_size < 0, original scale is used.
# voxel_size = -1

# # Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
# save_loss_log = True


# print("Apply Point-to-Point ICP")
# s = time.time()

# reg_point_to_point = o3d.t.pipelines.registration.icp(source_pcd, target_pcd, max_correspondence_distance,
#                               init_source_to_target, estimation, criteria,
#                               voxel_size, save_loss_log)

# icp_time = time.time() - s
# print("Time taken by Point-To-Point ICP: ", icp_time)
# print("Fitness: ", reg_point_to_point.fitness)
# print("Inlier RMSE: ", reg_point_to_point.inlier_rmse)
# print("trans : ", reg_point_to_point.transformation.numpy())



# voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])

# # List of Convergence-Criteria for Multi-Scale ICP:
# criteria_list = [
#     o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001,
#                                 relative_rmse=0.0001,
#                                 max_iteration=20),
#     o3d.t.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 15),
#     o3d.t.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 10)
# ]

# # `max_correspondence_distances` for Multi-Scale ICP (o3d.utility.DoubleVector):
# max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.14, 0.07])

# # Initial alignment or source to target transform.
# init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)

# # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
# estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()

# save_loss_log = False

# registration_ms_icp = o3d.t.pipelines.registration.multi_scale_icp(
#     source_pcd, target_pcd, voxel_sizes, criteria_list, 
#     max_correspondence_distances, init_source_to_target, estimation, save_loss_log
# )

# import numpy as np
# from typing import Dict, List, Any


# def avg_stats(stats:List):
#     avg = stats[0]
#     for e in avg:
#         if isinstance(avg[e], Dict):
#             this_stats = [stats[i][e] for i in range(len(stats))]
#             avg[e] = avg_stats(this_stats)
#         else:
#             avg[e] = np.mean([stats[i][e] for i in range(len(stats))])
#     return avg
        

# if __name__ == "__main__":
#     stats = []
#     for i in range(10):
#         this_stat = {"a0":{"a0b0": np.random.rand()*2+0.5}, "a1":np.random.rand()-0.5, "a2":{"a2b0":np.random.rand()+2.0}}
#         stats.append(this_stat)
#     print(stats)
#     print(avg_stats(stats))

# from sklearn.neighbors import KDTree
# import numpy as np

# X = np.random.random((10, 3))
# print(X)
# tree = KDTree(X)
# print(tree)
# dist, ind = tree.query(np.asarray([[0.5, 0.5, 0.5], [0.3, 0.2, 1.0]]), k=1)
# print(dist)
# print(np.asarray(ind).squeeze())

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
 
digits = datasets.load_digits(n_class=6)
X, y = digits.data, digits.target
n_samples, n_features = X.shape
 
'''显示原始数据'''
n = 20  # 每行20个数字，每列20个数字
img = np.zeros((10 * n, 10 * n))
for i in range(n):
    ix = 10 * i + 1
    for j in range(n):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
plt.figure(figsize=(8, 8))
plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.show()

'''t-SNE'''
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)
print(X.shape)
print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
 
'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

print(X_norm.shape)

plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()