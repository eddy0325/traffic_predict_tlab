# 学习路线

介绍**交通预测**领域方法演进（截至2024.12）

## 交通预测1.0

特点：主要考虑时间信息，较少考虑空间信息

* 基于统计学的模型：AR, MA, ARMA, ARIMA, SARIMA
* 基于机器学习的模型：SVR, GBDT
* 基于深度学习的模型：1d-CNN, LSTM, GRU

## 交通预测2.0

特点：同时建模时间和空间信息

推荐综述论文：《[DL-Traff: Survey and Benchmark of Deep Learning Models for Urban Traffic Prediction](https://dl.acm.org/doi/abs/10.1145/3459637.3482000)》

具体地，可粗略划分为以下两种类型

### A. Grid-based model

特点：将空间划分为网格，用CNN处理

经典论文：

1. ST-ResNet
   * 论文：《[Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/10735)》
   * 代码：https://github.com/BruceBinBoxing/ST-ResNet-Pytorch

### B. Graph-based model

特点：将空间抽象成图，用GNN处理

经典论文：

1. T-GCN
   * 论文：《[T-GCN: A temporal graph convolutional network for traffic prediction](https://ieeexplore.ieee.org/abstract/document/8809901)》
   * 代码：https://github.com/lehaifeng/T-GCN
2. A3T-GCN
   * 论文：《[A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting](https://www.mdpi.com/2220-9964/10/7/485)》
   * 代码：https://github.com/lehaifeng/T-GCN
3. STGCN
   * 论文：《[STGCN: A Spatial-Temporal Aware Graph Learning Method for POI Recommendation](https://ieeexplore.ieee.org/abstract/document/9338281)》
   * 代码：https://github.com/hazdzz/STGCN
4. DCRNN
   * 论文：《[DCRNN: A Deep Cross approach based on RNN for Partial Parameter Sharing in Multi-task Learning](https://arxiv.org/abs/2310.11777)》
   * 代码：https://github.com/chnsh/DCRNN_PyTorch
5. Graph WaveNet
   * 论文：《[Graph WaveNet for Deep Spatial-Temporal Graph Modeling](https://arxiv.org/abs/1906.00121)》
   * 代码：https://github.com/nnzhan/Graph-WaveNet
6. ASTGCN
   * 论文：《[AST-GCN: Attribute-Augmented Spatiotemporal Graph Convolutional Network for Traffic Forecasting](https://ieeexplore.ieee.org/abstract/document/9363197/)》
   * 代码：https://github.com/guoshnBJTU/ASTGCN-2019-pytorch
7. GMAN
   * 论文：《[GMAN: A Graph Multi-Attention Network for Traffic Prediction](https://aaai.org/ojs/index.php/AAAI/article/view/5477)》
   * 代码：https://github.com/VincLee8188/GMAN-PyTorch

## 交通预测3.0

特点：基于transformer的模型

这块不多介绍，[链接](https://blog.csdn.net/SmartLab307/article/details/129534937)中列举了6篇代表性论文及配套代码，自行阅读复现即可

附N篇基于transformer做通用预测模型的代表性论文供学习：
1. Informer
   * 论文：《[Informer: Beyond efficient transformer for long sequence time-series forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/17325)》
   * 代码：https://github.com/zhouhaoyi/Informer2020
