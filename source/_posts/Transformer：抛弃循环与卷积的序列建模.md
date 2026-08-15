---
title: Transformer：抛弃循环与卷积的序列建模
date: 2026-08-08
categories:
  - 论文研析
tags:
  - 经典论文
  - 人工智能基础
mathjax: true
---

本文是对 *Attention Is All You Need* 的精读笔记。本文按照论文的结构，依次梳理其研究动机，模型架构中的Encoder-Decoder、Self-Attention、Multi-Head Attention、三种 attention、前馈网络与位置编码、残差连接等，Self-Attention 优势论证，以及训练配置与实验结果，并梳理论文本身的结论与后续研究方向。

<!-- more -->

论文链接：

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

## 研究背景

在 Transformer 提出之前，序列建模的主流方法是 RNN，包括 LSTM、GRU 等变体。它们按照位置顺序递归计算隐藏状态 $h_t = f(h_{t-1}, x_t)$。

这种结构存在两个根本性缺陷。第一，无法并行计算，必须等前一步计算完成后才能继续计算下一步，因此训练效率较低。第二，长程依赖难以捕捉。两个距离较远的位置之间，信息需要经过多步传递，传播路径越长，信息越容易衰减。

Transformer 的核心思想是彻底舍弃循环和卷积结构，完全依靠 attention 机制建模序列中任意两个位置之间的依赖关系，从而同时解决并行计算和长程依赖这两个问题。

## 模型架构与核心原理

### Encoder-Decoder

### Self-Attention 机制

### Multi-Head Attention

### 前馈网络与位置编码

### 残差连接与 LayerNorm

## Self-Attention的优势



## 训练与实验结果

​       



## 总结

Transformer 是第一个完全基于注意力机制的序列模型，用 Multi-Head Self-Attention 替代了 encoder-decoder 架构中最常用的循环层。在 WMT 2014 英译德、英译法两个翻译任务上，Transformer 都取得了当时的最优结果，且训练速度远快于以往基于循环或卷积结构的模型。论文还通过英语成分句法分析任务的实验，验证了 Transformer 具备良好的任务泛化能力，即便在小数据量场景下也能取得有竞争力的结果，说明这一架构不局限于机器翻译。

作者在结论部分提到几个计划推进的方向：一是将 Transformer 拓展到文本以外的输入输出模态，如图像、音频、视频等；二是研究局部的、受限的注意力机制，以高效处理超长序列的输入输出；三是探索让生成过程更少依赖串行步骤的方法。这几个方向后来都在深度学习的发展中得到了印证：ViT 把 attention 用到了图像上，各类高效注意力针对长序列做了优化，扩散模型等非自回归生成方式减少串行依赖。
