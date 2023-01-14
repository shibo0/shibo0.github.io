---
title: Pattern Recognition - 第一章 绪论
key: 20230114
tags: Pattern-Recognition
mathjax: true
mathjax_autoNumber: true
---

模式识别(Pattern Recognition)学习笔记

`Pattern Recognition`{:.info} `Introduction`{:.info} `发展简史`{:.info} `分类器设计`{:.info} `模式识别系统`{:.info}

<!--more-->

## 1 模式识别

### 1.1 什么是模式识别

**模式识别**：使计算机模仿``人的感知能力``（人类智能），从``感知数据``（非结构化数据）中提取信息（判断物体和行为、现象，结构化知识）的过程。

**模式**：顾名思义，模式即为模式识别的对象，可以是一些文字，一个人，一张纸，鸟语花香等等一切可以进行识别测量的生活中的事物。

**人类智能**:\
    - 感知：视觉、听觉、触觉、嗅觉、味觉\
    - 学习：有教师学习，自主学习\
    - 思维：推理、回答问题、定理证明、下棋等\
    - 语言理解、对话\
    - 行为：表情、走路、运动\
**人工智能（ArtificialIntelligence，AI）**：构造智能机器（智能计算机、智能系统）的科学和工程，
    使机器模拟、延伸、扩展人类智能。
{:.warning}

### 1.2 模式识别应用

* 安全监控（身份识别、行为监控、交通监控）

* 空间探测与环境资源监测（卫星/航空遥感图像）

* 智能人机交互（表情、手势、声音、符号）

* 机器人环境感知（视听触觉）

* 人类健康（医学图像、体测数据）

* 工业应用（零部件/物品分类、损伤检测）

* 文档数字化（历史书籍、档案、手稿、标牌、票据等）

* 网络搜索、信息提取和过滤（文本、图像、视频、音频、多媒体 ）

* 舆情、疫情分析（互联网、大数据）


模式是被研究主要集中在两个方面，一是研究生物体是如何感知对象的，属于认知科学范畴，二是在给定的任务下，如何用计算机实线模式识别的理论和方法。
{:.info}

## 2 模式识别发展简史

### 2.1 早期模式识别技术

1914年，以色列发明家Emanuel Godlberg开发了一个阅读字符并转化为电报码的机器。奥地利工程师Gustav Tauschek发明的OCR机器Reading Machine于1929年获得德国专利

>O.G. Selfridge, Pattern recognition and modern computers, Proc. Western Joint Computer Conference, 1955.
>C.K. Chow, An optimum character recognition system using decision functions, IRE Trans. Electronic Computers, EC-6(4): 247-254, 1957.


### 2.2 早期代表性工作和事件

* Nils J. Nilsson, Learning Machines: Foundations of Trainable PatternClassifying Systems, McGraw-Hill Book Company, 1965.

* Pattern Recognition 1966 IEEE Workshop

* G. Nagy, State of the art in pattern recognition, Proc. IEEE, 1968.

* K. S. Fu, Sequential Methods in Pattern Recognition and Machine Learning,1968

* Pattern Recognition Journal, 1968

* Textbooks

  * Keinoske Fukunaga, Introduction to Statistical Pattern Recognition, First edition, 1972. (2nd edition, 1990)

  * Richard O. Duda, Peter E. Hart, Pattern Classification and Scene Analysis,1973. (2nd edition, 2001)


* 句法模式识别
  * K. S. Fu, Syntactic Methods in Pattern Recognition. New York: Academic, 1974.

* 国际模式识别大会IJCPR 1973,1974,1976,1978, ICPR from 1980

* 国际模式识别学会IAPR: 1974年IJCPR期间开始筹建，1976年IJCPR期间召开了第一次执委会会议，1977年开始接受会员申请， 在1978年IJCPR期间宣告正式成立

* IEEE T-PAMI, 1978

* 80年代

  * 多层神经网络，BP算法

    * D.E. Rumelhart, G.E. Hinton, R.J. Williams, Learning internal representation by error propagation, in Parallel Distributed Processing, vol.1: Foundations, MIT Press, 1986.

    * Paul Werbos, PhD Thesis, 1974.

  * 卷积神经网络最早出现于1989年

    * Yann LeCun, et al. Handwritten digit recognition with a back-propagation network. NIPS 1989. (Fukushima, Neocognitron, 1980)

* 90年代：多种学习方法兴起

  * 支持向量机(SVM)

    * C. Cortes and V. Vapnik. Support vector networks. Machine Learning, 20:273-297, 1995.

  * 多分类器系统(Ensemble)

  * 半监督学习

  * 多标签学习

  * 多任务学习

* 21世纪初

  * 概率图模型

    * 马尔科夫随机场(MRF)

    * 隐马尔科夫模型(HMM)：80年代开始用于语音识别，90年代开始用于手写文本识别

    * 条件随机场(CRF)：L. Lafferty, et al., ICML 2001.

  * 迁移学习

  * 深度学习

### 2.3 模式识别主要方法和事件演化图

<center>
<img src="https://img-blog.csdnimg.cn/fc952618a5a04ce3a0a9afd490622dfc.png"
      alt = "主要方法和事件演化图"/></center>
<center><p>图1 主要方法和事件演化图</p></center>


### 2.4 与其他领域的关系

<center>
<img src="https://img-blog.csdnimg.cn/44d6467da5ce4c00bfe17861194ff601.png"
      alt = "各领域关系" style="zoom:60%"/></center>
<center><p>图2 与其他领域的关系: 模式识别-机器学习-数据挖掘-计算机视觉</p></center>


## 3 模式识别系统


**传感器**

一个模式分类系统的输入通常取自一些传感器，并且依赖于传感器的特性和局限性。
{:.info}

**分割和组织**

分割问题可谓模式识别中的最深层的问题之一，这在自动语言识别、图像物体分割任务中比较常见。然而如今深度学习端到端（End-to-End）的技术已经不需要再进行手动处理。
{:.info}

**特征处理**

特征提取器通常要提取具有如下性质的特征描述，即，来自同一类别的不同样本的特征值应该非常相近，而来自不同类别的样本的特征值应该有很大的差异。
{:.info}

**分类器**

系统中的分类器的作用是：根据特征提取器得到的特征向量来给一个被测对象赋一个类别标记。分类的难易程度取决于两个因素，其一是来自同一个类别的不同个体之间的特征值的波动，其二是属于不同类别的样本的特征值之间的差异。
{:.info}

**后处理**

后处理是分析分类器的输出结果，比如寻找具有最低分类误差率的分类器，或者降低总体的代价（风险）。
{:.info}


整个系统的处理过程是非端到端的，而端到端的技术省去了分步骤的学习以及在每一个独立学习任务执行之前所做的数据标注，为样本做标注的代价是昂贵的、易出错的。端到端学习指的是深度学习模型中，所有参数或原先几个步骤需要确定的参数被联合学习，而不是分步骤学习。
{:.warning}

## 4 模式识别设计

### 4.1 分类器训练（学习）过程

[//]: # ![分类器训练过程](https://img-blog.csdnimg.cn/b4488de11c1c45f4a2701b13b678e6ab.png){:width="512px"}
<center>
<img src="https://img-blog.csdnimg.cn/b4488de11c1c45f4a2701b13b678e6ab.png"
      alt = "分类器训练过程" style="zoom:50%"/></center>
<center><p>图3 分类器训练过程</p></center>


### 4.2 分类器训练-评价过程

训练和测试过程分开，使用不同的样本集，应用测试集评估模型性能。

### 4.3 模型选择和评价

**模型选择(model selection)**有两层含义：一是在假设空间上训练得到的模型可能不止一个，需要从中进行选择；二是对于一个具体问题，我们可能希望尝试不同方法，于是就有了不同的模型，在这些模型训练结束后，我们需要决定使用哪一个，但这种模型选择往往需要结合模型评估方法，因为对于某种归纳偏好，不同方法下的不同模型的实现各不相同，只能根据在测试集上的最终表现效果来选择。

**模型评估(model assessment)**是指对于一种具体方法输出的最终模型，使用一些指标和方法来评价它的泛化能力。

### 4.4 数据划分方式

* **留出法 holdout**

  * 性能评估（Performance evaluation）：将数据划分成训练集和测试集

  * 模型选择（Model selection）：将训练集划分成估计（训练）集和验证集

<center>
<img src="https://img-blog.csdnimg.cn/6b6d75d5209a4974b4b8d3d42f468a8f.png"
      alt = "留出法" style="zoom:60%"/></center>
<center><p>图4 留出法</p></center>

* **交叉验证法 Cross-validation**

  * 将数据集划分成N等份，每等份轮流做Test，其余部分用于训练

  * 留一法 Leave-one-out(LOO):交叉验证的一个特例，留一法使用的训练集与初始数据集相比只少了一个样本，这就使得留一法中被实际评估的模型与期望评估的模型很相似。因此，留一法的评估结果往往被认为比较准确，但是缺陷就是碰到大数据集时，计算开销时无法接受。

* **自助法 Bootstrapping**：自助法就是利用有限的样本经由多次重复抽样，建立起足以代表母体样本分布之新样本，在机器学习中解决了样本不足的问题。

[//]: # ![自助法](https://img-blog.csdnimg.cn/eea0340de629444d9cb613de14155d57.jpeg)
<center>
<img src="https://img-blog.csdnimg.cn/eea0340de629444d9cb613de14155d57.jpeg"
      alt = "自助法" style="zoom:60%"/></center>
<center><p>图5 自助法</p></center>

### 4.5 泛化性能

**泛化性能(Generalization Performance)**：测试数据上的分类性能

**过拟合/过学习**：用复杂分类器将训练数据分类错误率降到极低。通常情况下，训练数据越多、越有代表性，则泛化性能越好

[//]: # ![泛化性能](https://img-blog.csdnimg.cn/f65c7085a4e34791bbc2241dc4c272bb.png)
<center>
<img src="https://img-blog.csdnimg.cn/f65c7085a4e34791bbc2241dc4c272bb.png"
      alt = "泛化性能，错误率随训练集尺寸变化" style="zoom:70%"/></center>
<center><p>图6 泛化性能，错误率随训练集尺寸变化</p></center>

**分类器（模型）复杂度对泛化性能的影响：**

• 训练数据不变的情况下，分类器越复杂，对训练数据拟合程度越高

• 过拟合情况下，泛化性能会下降

[//]: # ![请添加图片描述](https://img-blog.csdnimg.cn/882b40f500c841daae5b59799ba6f710.png)

[//]: # ![请添加图片描述](https://img-blog.csdnimg.cn/482f4d25e64f4b3aad0ce7f8cbee74aa.png)

<div class="grid-container">
<div class="grid grid--p-3">
<div class="cell cell--12 cell--md-11 cell--lg-4" markdown="1">
![Image](https://img-blog.csdnimg.cn/882b40f500c841daae5b59799ba6f710.png "Image_rounded"){:.rounded}
</div>
<div class="cell cell--12 cell--md-11 cell--lg-4" markdown="1">
![Image](https://img-blog.csdnimg.cn/482f4d25e64f4b3aad0ce7f8cbee74aa.png "Image_circle+shadow"){:.rounded}
</div>
</div>
</div>

## 5 模式识别方法分类

### 5.1 按照模式/模型表示方式分类

<center>
<img src="https://img-blog.csdnimg.cn/1232598ec9654d3281a452465c841fef.png"
      alt = "" style="zoom:70%"/></center>

[//]:#![请添加图片描述](https://img-blog.csdnimg.cn/1232598ec9654d3281a452465c841fef.png)

### 5.2 按照学习方法分类

**监督(Supervised)学习**：训练样本有类别标号

**无监督(Unsupervised)学习**：训练样本无类别标号，得到数据结构表示或分布

**半监督(Semi-supervised)学习**：训练样本一部分有类别标号，一部分没有

**Reinforcement Learning**：学习过程中给出奖惩信号，例如，Deep Mind（被Google收购）基于深度神经网络强化学习的玩视频游戏程序

**Domain Adaptation**：测试样本分布发生变化，分类器参数自适应

**Online (Incremental) Learning**：数据顺序出现（且过去数据不能保存），学新不忘旧

### 5.3 生成/判别模型

**生成(Generative)模型**：表示各个类别内部结构或特征分布$p(x\|c)$

**判别(Discriminative)模型**：表示不同类别之间的区别，一般为判别
函数(Discriminant function)、边界函数或后验概率$P(c|x)$
