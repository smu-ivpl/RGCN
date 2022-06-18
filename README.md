# Residual-Based Graph Convolutional Network for Emotion Recognition in Conversation for Smart Internet of Things
#### Young-Ju Choi, Young-Woon Lee, and Byung-Gyu Kim
#### Intelligent Vision Processing Lab. (IVPL), Sookmyung Women's University, Seoul, Republic of Korea
----------------------------
#### This repository is the official PyTorch implementation of the paper published in _Big Data, Mary Ann Liebert, Inc., publishers_.
[![paper](https://img.shields.io/badge/paper-PDF-<COLOR>.svg)](https://www.liebertpub.com/doi/pdf/10.1089/big.2020.0274)

----------------------------
## Summary of paper
#### Abstract
> _Recently, emotion recognition in conversation (ERC) has become more crucial in the development of diverse Internet of Things devices, especially closely connected with users. The majority of deep learning-based methods for ERC combine the multilayer, bidirectional, recurrent feature extractor and the attention module to extract sequential features. In addition to this, the latest model utilizes speaker information and the relationship between utterances through the graph network. However, before the input is fed into the bidirectional recurrent module, detailed intrautterance features should be obtained without variation of characteristics. In this article, we propose a residual-based graph convolution network (RGCN) and a new loss function. Our RGCN contains the residual network (ResNet)-based, intrautterance feature extractor and the GCN-based, interutterance feature extractor to fully exploit the intra–inter informative features. In the intrautterance feature extractor based on ResNet, the elaborate context feature for each independent utterance can be produced. Then, the condensed feature can be obtained through an additional GCN-based, interutterance feature extractor with the neighboring associated features for a conversation. The proposed loss function reflects the edge weight to improve effectiveness. Experimental results demonstrate that the proposed method achieves superior performance compared with state-of-the-art methods._
>

#### Network Architecture
<p align="center">
  <img width="900" src="./images/fig3.PNG">
</p>

<p align="center">
  <img width="900" src="./images/fig4.PNG">
</p>

<p align="center">
  <img width="900" src="./images/fig5.PNG">
</p>

#### Experimental Results
<p align="center">
  <img width="900" src="./images/table4.PNG">
</p>

<p align="center">
  <img width="900" src="./images/table5.PNG">
</p>
