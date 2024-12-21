# 学习路线

@ 维护人：励英迪

@ 联系方式：yingdi.li@qq.com

本文档介绍**交通预测**领域方法演进（截至2024.12）

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
   * 赏析：本质上是基于GRU的框架，用GCN代替了GRU cell中的线性层；ConvLSTM也是一样的套路
2. A3T-GCN
   * 论文：《[A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting](https://www.mdpi.com/2220-9964/10/7/485)》
   * 代码：https://github.com/lehaifeng/T-GCN
   * 赏析：T-GCN的衍生，输出模块用了attention
3. STGCN
   * 论文：《[Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875)》
   * 代码：https://github.com/hazdzz/STGCN
   * 赏析：xxx
4. DCRNN
   * 论文：《[Diffusion convolutional recurrent neural network: Data-driven traffic forecasting](https://arxiv.org/abs/1707.01926)》
   * 代码：https://github.com/chnsh/DCRNN_PyTorch
   * 赏析：和T-GCN一样的框架，用扩散卷积代替了GRU cell中的线性层；多步预测的时候采用seq2seq的范式逐个生成每个时刻的预测结果，encoder的last hidden state用来初始化decoder的hidden state
5. Graph WaveNet
   * 论文：《[Graph WaveNet for Deep Spatial-Temporal Graph Modeling](https://arxiv.org/abs/1906.00121)》
   * 代码：https://github.com/nnzhan/Graph-WaveNet
   * 赏析：时空分步处理，时间用gated TCN，空间用考虑自适应邻接矩阵的增强版扩散卷积；多步预测的时候先把信息汇总到最后一个时间片，然后用MLP直接生成所有时刻预测结果
6. ASTGCN
   * 论文：《[Attention based spatial-temporal graph convolutional networks for traffic flow forecasting](http://ojs.aaai.org/index.php/AAAI/article/view/3881)》
   * 代码：https://github.com/guoshnBJTU/ASTGCN-2019-pytorch
   * 赏析：xxx
7. GMAN
   * 论文：《[GMAN: A Graph Multi-Attention Network for Traffic Prediction](https://aaai.org/ojs/index.php/AAAI/article/view/5477)》
   * 代码：https://github.com/VincLee8188/GMAN-PyTorch
   * 赏析：纯基于注意力的模型，没有使用传统的时空特征提取模块；非常优雅，推荐阅读

## 交通预测3.0

特点：基于transformer的模型

可学习[链接](https://blog.csdn.net/SmartLab307/article/details/129534937)中列举的6篇代表性论文：

1. STTN
   * 论文：《[Spatial-Temporal Transformer Networks for Traffic Flow Forecasting](https://arxiv.org/abs/2001.02908)》
   * 代码：https://github.com/wubin5/STTN
   * 赏析：xxx
2. Traffic Transformer
   * 论文：《[Learning dynamic and hierarchical traffic spatiotemporal features with Transformer](https://ieeexplore.ieee.org/document/9520129)》
   * 代码：https://github.com/R0oup1iao/Traffic-Transformer
   * 赏析：xxx
3. ASTGNN
   * 论文：《[Learning Dynamics and Heterogeneity of Spatial-Temporal Graph Data for Traffic Forecasting](https://ieeexplore.ieee.org/document/9346058)》
   * 代码：https://github.com/guoshnBJTU/ASTGNN
   * 赏析：xxx
4. MGT
   * 论文：《[Meta Graph Transformer: A Novel Framework for Spatial–Temporal Traffic Prediction](https://www.sciencedirect.com/science/article/pii/S0925231221018725)》
   * 代码：https://github.com/lonicera-yx/MGT
   * 赏析：xxx
5. ASTTN
   * 论文：《[Adaptive Graph Spatial-Temporal Transformer Network for Traffic Flow Forecasting](https://arxiv.org/abs/2207.05064)》
   * 代码：https://github.com/yokifly/ASTTN_pytorch
   * 赏析：xxx
6. PDFormer
   * 论文：《[PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/25556)》
   * 代码：https://github.com/BUAABIGSCity/PDFormer/tree/master
   * 赏析：xxx

附N篇基于transformer做通用预测模型的代表性论文供学习：
1. Informer
   * 论文：《[Informer: Beyond efficient transformer for long sequence time-series forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/17325)》
   * 代码：https://github.com/zhouhaoyi/Informer2020

## OD预测支线

1. GEML
   * 论文：《[Origin-Destination Matrix Prediction via Graph Convolution: a New Perspective of Passenger Demand Modeling](https://dl.acm.org/doi/abs/10.1145/3292500.3330877)》
   * 代码：https://github.com/gonglucky/GEML-Origin-Destination-Matrix-Prediction-via-Graph-Convolution
2. DNEAT
   * 论文：《[DNEAT: A novel dynamic node-edge attention network for origin-destination demand prediction](https://www.sciencedirect.com/science/article/pii/S0968090X20307518)》
   * 代码：暂无
3. CMOD
   * 论文：《[Continuous-Time and Multi-Level Graph Representation Learning for Origin-Destination Demand Prediction](https://dl.acm.org/doi/abs/10.1145/3534678.3539273)》
   * 代码：https://github.com/liangzhehan/CMOD
4. HMOD
   * 论文：《[Dynamic Graph Learning Based on Hierarchical Memory for Origin-Destination Demand Prediction](https://arxiv.org/abs/2205.14593)》
   * 代码：https://github.com/Rising0321/HMOD
