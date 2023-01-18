---
title: Pattern Recognition - 第二章 贝叶斯决策论
key: 20230118
tags: Pattern-Recognition
mathjax: true
mathjax_autoNumber: true
---

模式识别(Pattern Recognition)学习笔记

`Pattern Recognition`{:.info} `Bayes decision`{:.info} `最小风险决策`{:.info} `判别函数和判别面`{:.info} `朴素贝叶斯`{:.info}

<!--more-->

贝叶斯决策论是解决模式分类问题的一种基本统计途径。其出发点是利用概率的不同分类决策与相应的决策代价之间的定量折中。它作了如下的假设，即决策问题可以用概率的形式来描述，并且假设所有有关的概率结构均已知。

## 1 引言

**类别** $\omega_i$, $i=1,...,c$

**特征矢量** $x=[x_1,...,x_d] \in R^d$

**先验概率**是指根据以往经验和分析得到的概率。 $P(\omega_i)$, $\sum_{i=1}^c P(\omega_i)=1$

**概率密度函数（条件概率）**，即类别状态为$\omega$时的$x$的概率密度函数 $p(x\|\omega_i)$, 也称为似然


**后验概率**$P(\omega_i\|x)$：事情已经发生，要求这件事情发生的原因是由某个因素引起的可能性的大小。

$$贝叶斯公式：P(\omega_i|x)=\frac{p(x|\omega_i)P(\omega_i)}{p(x)}=\frac{p(x|\omega_i)P(\omega_i)}{\sum_{j=1}^c p(x|\omega_j)P(\omega_j)}$$

$$\sum_{i=1}^c P(\omega_i|x) = 1$$

贝叶斯公式用非正式的英语表示：

$$posterior=\frac{likelihood \times prior}{evidence}$$


## 2 最小风险决策：贝叶斯决策的一般形式

**决策代价（loss，cost）**

- 将正确类别$\omega_j$, 决策为$\alpha_i$的风险（代价）$\lambda_{ij}=\lambda(\alpha_i\|\omega_j)$,

- 有时$\lambda_{ij}$和$\lambda_{ji}$相差很大，比如医疗诊断、工业检测等场合

**条件风险 Condition risk**

$$R(\alpha_i|x)=\sum_{j=1}^c \lambda(\alpha_i|\omega_j)P(\omega_j|x)$$

**Overall(Expected) risk**

$$R=\int R(\alpha(x)|x)p(x) dx,\ \alpha(x) \in \{\alpha_1,...,\alpha_c\}$$

**Minimum risk decision(Bayes decision)**

$$\mathop{\arg\min}\limits_{i} R(\alpha_i|x)$$


## 3 带拒识的决策

对于$C+1$类

$$
\begin{equation}
\lambda(\alpha_i|\omega_j)=
=\left\{
	\begin{array}{ll}
		0, & i=j \\
		\lambda_s, & i \neq j \\
    \lambda_r, & reject \ \ \ \lambda_r < \lambda_s
	\end{array}\right.
\end{equation}
$$

风险函数$R(\alpha_i\|x)=\sum_{j=1}^c \lambda(\alpha_i\|\omega_j)P(\omega_j\|x)$=>

$$
\begin{equation}
R_i(x)=
=\left\{
	\begin{array}{ll}
		\lambda_s[1-P(\omega_i|x)], & i=1,...,c \\
		\lambda_r, & reject \\
	\end{array}\right.
\end{equation}
$$

**最小化风险函数**

$$
\begin{equation}
\mathop{\arg\min}\limits_{i} R_i(x)=
=\left\{
	\begin{array}{ll}
		\mathop{\arg\max}\limits_{i} P(\omega_i|x), & if\ \mathop{\max}\limits_{i} P(\omega_i|x)>1-\frac{\lambda_r}{\lambda_s} \\
		reject, & otherwise \\
	\end{array}\right.
\end{equation}
$$

符合人的直觉经验：最大后验概率比较大，看得比较准，就预测该类别；反之，拒识。

## 4 判别函数、决策面

### 4.1 判别函数(Discriminant Function)

– 表征模式属于每一类的广义似然度$g_i(x)$, $i=1,…,c$

– 分类决策 $$\mathop{\arg\max}\limits_{i} g_i(x)$$

– 贝叶斯判别函数$g_i(x)$可以为$g_i(x)=-R(\alpha_i\|x)$, $g_i(x)=P(\omega_i\|x)$, $g_i(x)=p(x\|\omega_i)P(\omega_i)$ or $g_i(x)=log\ p(x\|\omega_i)+log\ P(\omega_i)$

高斯密度下的判别函数
\
$$g_i(x)=ln\ p(x|\omega_i)+ln\ P(\omega_i)$$
\
$$多变量密度函数p(x|\omega_i)=\frac{1}{(2\pi)^{d/2}\left| \Sigma_i \right|^{1/2}}\ \ exp[-\frac{1}{2}(x-\mu_i)\Sigma_i^{-1}(x-\mu_i)]$$
\
$$g_i(x)=-\frac{1}{2}(x-\mu_i)^t\Sigma_i^{-1}(x-\mu_i)-\frac{d}{2}ln\ 2\pi-\frac{1}{2}ln\ \left| \Sigma_i \right| + ln\ P(\omega_i)$$
\
在不同covariance假设条件下得到一些特殊形式
{:.warning}

### 4.2 决策面(Decision surface)

决策面：特征空间中二类判别函数相等的点的集合，例如决策面$g(x)=g_1(x)-g_2(x)=0$

### 4.3 多类情况

有很多种方式来表述模式分类器，其中用的最多的是一种判别函数$g_i(x)$,$i=1,...,c$的形式，如果对于所有的$j \neq i$，有

$$g_i(x)>g_j(x)$$

则分类器将这个特征向量$x$判为$\omega_i$。因此，此分类器可视为一个计算c个判别函数并选取与最大判别值对应的类别的网络或机器。

## 5 贝叶斯决策用于模式分类

* Bayes决策的关键

  * 类条件概率密度估计

  * 先验概率：从训练样本估计或假设等概率

  * 决策代价$λ_{ij}$，一般为0-1代价

* 分类器设计

  * 收集训练样本

  * 用每一类的样本估计**类条件概率密度**$p(x\|\omega_i)$

  * 估计类先验概率

  * 模型参数集：$\\{p(x\|\omega_i,\theta_i),P(\omega_i)\\}$, $i=1,...,c$

* 分类过程

  * 计算测试样本$x$属于每一类的后验概率

  * 最大后验概率/最小风险决策

  特征向量x中的元素为离散数值时，贝叶斯公式中的概率密度函数p由概率分布函数P代替，条件风险的定义不变，贝叶斯决策论的判决规则不变，通过最大化后验概率来最小化误差概率的基本原则也不变。
  {:.warning}

## 6 概率密度估计方法

* 参数法：假定概率密度函数形式

  * Distribution: Gaussian,Gamma,Bernouli

  * Parameter estimation: maximum-likelihood (ML), Bayesian estimation

* 非参数法：可以表示任意概率分布，无函数形式

  * Parzen window, k-NN

  * 需要保存所有或大部分样本

* Semi-parametric: 近似任意概率分布，有函数形式

  * Distribution: Gaussian mixture (GM)

  * Estimation: expectation-maximization (EM)

## 7 朴素贝叶斯分类器

贝叶斯分类器属于有监督学习，它需要标签化的训练数据集进行概率计算；所谓``朴素``，是假定所有输入事件之间是相互独立。进行这个假设是因为独立事件间的概率计算更简单。

**朴素贝叶斯分类原理**

$$P(\omega_i|X)=\frac{P(\omega_i)p(X|\omega_i)}{P(X)}$$

其中$\omega_i$为类别i，$X=\\{x_1,...,x_n\\}$为特征向量，$x_j$为特征向量的分量

假设$x_i$之间都是相互独立的，可以推导出：

$$p(X|\omega_i)=\prod \limits_{j=1}^n p(x_j|\omega_i)$$

从而将原始的贝叶斯公式简化为：

$$P(\omega_i|X)=\frac{P(\omega_i) \prod \limits_{j=1}^n p(x_j|\omega_i)}{P(X)}$$

对于计算，$P(X)$表示该样本出现的概率，在计算中认为是常数，可以忽略不计，故而求**最大后验估计**问题可以简化为求**极大似然估计**问题，即

$$P(\omega_i|X)\ \propto\ P(\omega_i)\prod \limits_{j=1}^n P(x_j|\omega_i)$$

利用这个分类规则依次计算待判别样本属于全部分类类别的概率值，得到最大的值：

$$\mathop{\arg\max}\limits_{i} P(\omega_i)\prod \limits_{j=1}^n P(x_j|\omega_i)$$
