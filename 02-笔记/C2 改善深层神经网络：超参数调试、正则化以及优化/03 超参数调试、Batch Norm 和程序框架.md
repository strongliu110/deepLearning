# 超参数调试、Batch Norm 和程序框架

[TOC]

## 3.1 调试处理

深度神经网络需要调试的超参数（Hyperparameters）较多，包括：

- $\alpha $：学习因子（**最重要，需要重点调试**，优先级1）
- $\beta $：动量梯度下降因子（优先级2）
- $\beta _{1},\beta _{2},\epsilon $：Adam算法参数（一般常设置为0.9，0.999和$10^{-8}$，不需要反复调试）
- \#layers：神经网络层数（优先级3）
- \#hidden units：各隐藏层神经元个数（优先级2）
- learning rate decay：学习因子下降参数（优先级3）
- mini-batch size：批量训练样本包含的样本个数（优先级2）

如何选择和调试超参数？传统的机器学习中，会使用**网格搜索（ grid search）**来选择最优超参数。但是在深度神经网络模型中，我们一般不采用这种均匀间隔取点的方法，比较好的做法是使用随机选择。**随机化选择参数的目的是为了尽可能地得到更多种参数组合**。

例如有两个待调试的参数，分别在每个参数上选取5个点，如果使用均匀采样的话，每个参数只有5种情况；而使用随机采样的话，每个参数有25种可能的情况，因此更有可能得到最佳的参数组合。

这种做法带来的另外一个好处就是**对重要性不同的参数之间的选择效果更好**。假设hyperparameter1为$\alpha $，hyperparameter2为$\epsilon$，显然二者的重要性是不一样的。如果使用第一种均匀采样的方法，$\epsilon$的影响很小，相当于只选择了5个$\alpha $值。而如果使用第二种随机采样的方法，$\epsilon$和$\alpha $都有可能选择25种不同值。这大大增加了$\alpha $调试的个数，更有可能选择到最优值。其实，在实际应用中完全不知道哪个参数更加重要的情况下，随机采样的方式能有效解决这一问题，但是均匀采样做不到这点。

在经过随机采样之后，我们可能得到某些区域模型的表现较好。然而，为了得到更精确的最佳参数，我们应该继续对选定的区域进行**由粗到细的采样（coarse to fine sampling scheme）**。也就是放大表现较好的区域，再对此区域做更密集的随机采样。例如，对下图中右下角的方形区域再做25点的随机采样，以获得最佳参数。

![随机采样](http://img.blog.csdn.net/20171031230142816?)

## 3.1 为超参数选择合适的范围

**使用随机采样，对于某些超参数是可以进行尺度均匀采样的，但是某些超参数需要选择不同的合适尺度进行随机采样**。

例如对于超参数#layers和#hidden units，都是正整数，是可以进行均匀随机采样的，即超参数每次变化的尺度都是一致的（如每次变化为1，犹如一个刻度尺一样，刻度是均匀的）。

但是，对于某些超参数，可能需要非均匀随机采样（即非均匀刻度尺）。例如超参数$\alpha $，待调范围是[0.0001, 1]。如果使用均匀随机采样，那么有90%的采样点分布在[0.1, 1]之间，只有10%分布在[0.0001, 0.1]之间。这在实际应用中是不太好的，因为最佳的$\alpha $值可能主要分布在[0.0001, 0.1]之间，而[0.1, 1]范围内$\alpha $值效果并不好。因此我们更关注的是区间[0.0001, 0.1]，应该在这个区间内细分更多刻度。

**通常的做法是将linear scale转换为log scale，将均匀尺度转化为非均匀尺度，然后再在log scale下进行均匀采样**。

![尺度变换](http://img.blog.csdn.net/20171101094754376?)

一般解法是，如果线性区间为[a, b]，令m=log(a)，n=log(b)，则对应的log区间为[m,n]。对log区间的[m,n]进行随机均匀采样，然后得到的采样值r，最后反推到线性区间，即$10^{r}$。$10^{r}$就是最终采样的超参数。相应的Python语句为：

```python
m = np.log10(a)
n = np.log10(b)
r = np.random.rand()
r = m + (n-m)*r
r = np.power(10,r)
```

除了$\alpha$之外，动量梯度因子$\beta $也是一样，在超参数调试的时候也需要进行非均匀采样。一般$\beta $的取值范围在[0.9, 0.999]之间，那么$1-\beta $的取值范围就在[0.001, 0.1]之间。那么**直接对1−β在[0.001, 0.1]区间内进行log变换即可**。

这里解释下为什么$\beta $也需要向$\alpha $那样做非均匀采样。假设$\beta $从0.9000变化为0.9005，那么$\frac{1}{1-\beta }$基本没有变化。但假设$\beta $从0.9990变化为0.9995，那么$\frac{1}{1-\beta }$前后差别1000。$\beta $越接近1，指数加权平均的个数越多，变化越大。所以对$\beta $接近1的区间，应该采集得更密集一些。

## 3.3 超参数训练的实践：Pandas VS Caviar

经过调试选择完最佳的超参数并不是一成不变的，一段时间之后（例如一个月），**需要根据新的数据和实际情况，再次调试超参数，以获得实时的最佳模型**。

**深度学习的两种训练方式**:

1. **单模型养成式 (Babysitting one model)**，即熊猫式（Panda approach）: 在计算资源有限的情况下, 难于同时训练多个模型, 因此一般精心地训练一个模型。
2. **多模型并行式 (Training many models in parallel)**，即鱼子式（Caviar approach）: 在计算资源充足的情况下, 可以同时训练多个模型, 最后选择最优的即可。

一般来说，对于非常复杂或者数据量很大的模型，使用熊猫式更多一些。

## 3.4 标准化网络的激活函数

Batch Normalization不仅可以让调试超参数更加简单，而且可以让神经网络模型更加“健壮”。也就是说较好模型可接受的超参数范围更大一些，包容性更强，使得更容易去训练一个深度神经网络。

我们已经知道对样本输入进行标准化 (减去样本均值，除以样本方差) 能加速学习。

其实在神经网络中，第l层隐藏层的输入就是第l−1层隐藏层的输出$A^{\left[ l-1 \right]}$。对$A^{\left[ l-1 \right]}$进行标准化处理，从原理上来说可以提高$W^{\left[ l \right]}$和$b^{\left[ l \right]}$的训练速度和准确度。这种**对各隐藏层的标准化处理就是Batch Normalization**。值得注意的是，实际应用中，一般是对$Z^{\left[ l-1 \right]}$进行标准化处理而不是$A^{\left[ l-1 \right]}$，其实差别不是很大。

Batch Normalization对第l层隐藏层的输入$Z^{\left[ l-1 \right]}$做如下标准化处理，忽略上标[l−1]：

$\mu =\frac{1}{m}\sum_{i}^{m}{z^{\left( i \right)}}$

$\sigma ^{2}=\frac{1}{m}\sum_{i}^{m}{\left( z^{\left( i \right)}-\mu  \right)^{2}}$

$z_{norm}^{\left( i \right)}=\frac{z^{\left( i \right)}-\mu }{\sqrt{\sigma ^{2}+\epsilon }}$

其中，m是单个mini-batch包含样本个数，$\epsilon $是为了防止分母为零，可取值$10^{-8}$。这样，使得该隐藏层的所有输入$z^{\left( i \right)}$均值为0，方差为1。

但是，**大部分情况下并不希望所有的$z^{\left( i \right)}$均值都为0，方差都为1**，也不太合理。因此还需要对$z^{\left( i \right)}$进行进一步处理：

$\tilde{z}^{\left( i \right)}=\gamma \cdot z_{norm}^{\left( i \right)}+\beta $

上式中，$\gamma $和$\beta$是可学习参数，类似于W和b一样，可以通过梯度下降等算法求得。这里，$\gamma $和$\beta$的作用是让$\tilde{z}^{\left( i \right)}$的均值和方差为任意值，只需调整其值就可以了。例如，令：$\gamma =\sqrt{\sigma ^{2}+\epsilon },\beta =\mu $，则$\tilde{z}^{\left( i \right)}=z^i $，即identity function。可见，设置γ和β为不同的值，可以得到任意的均值和方差。

这样，通过Batch Normalization，对隐藏层的各个$z^{\left[ l \right]\left( i \right)}$进行标准化处理，得到$\tilde{z}^{\left[ l \right]\left( i \right)}$，替代$z^{\left[ l \right]\left( i \right)}$。

值得注意的是，输入的标准化处理（Normalizing inputs）和隐藏层的标准化处理（Batch Normalization）是有区别的。Normalizing inputs使所有输入的均值为0，方差为1。而Batch Normalization可使各隐藏层输入的均值和方差为任意值。实际上，**从激活函数的角度来说，如果各隐藏层的输入均值在靠近0的区域即处于激活函数的线性区域，这样不利于训练好的非线性神经网络，得到的模型效果也不会太好**。这也解释了为什么需要用$\gamma $和$\beta$来对$z^{\left[ l \right]\left( i \right)}$作进一步处理。

## 3.5 将 Batch Norm 拟合进神经网络

我们已经知道了如何对某单一隐藏层的所有神经元进行Batch Norm，接下来将研究如何把Bath Norm应用到整个神经网络中。

对于L层神经网络，经过Batch Norm的作用，整体流程如下：

![Batch Norm](http://img.blog.csdn.net/20171102090304433?)

实际上，Batch Norm经常使用在mini-batch上，这也是其名称的由来。

值得注意的是，因为Batch Norm对各隐藏层$Z^{\left[ l \right]}=W^{\left[ l \right]}A^{\left[ l-1 \right]}+b^{\left[ l \right]}$都有去均值的操作，所以这里的常数项$b^{\left[ l \right]}$可以消去，其数值效果完全可以由$\tilde{z}^{\left( l \right)}$中的$\beta $来实现。因此，**在使用Batch Norm的时候，可以忽略各隐藏层的常数项$b^{\left[ l \right]}$。在使用梯度下降算法时，分别对$W^{\left[ l \right]}$，$\beta ^{\left[ l \right]}$，$\gamma^{\left[ l \right]}$进行迭代更新**。除了传统的梯度下降算法之外，还可以使用我们之前介绍过的动量梯度下降、RMSprop或者Adam等优化算法。

## 3.6 Batch Norm 为什么有效？

如果实际应用的样本与训练样本分布不同，即发生了**样本点变化（covariate shift）**，则一般是要对模型重新进行训练的。**而Batch Norm的作用恰恰是减小covariate shift的影响，让模型变得更加健壮，鲁棒性更强**。Batch Norm减少了各层$W^{\left[ l \right]}$、$b^{\left[ l \right]}$之间的耦合性，让各层更加独立，实现自我训练学习的效果。也就是说，如果输入发生covariate shift，那么因为Batch Norm的作用，对个隐藏层输出$Z^{\left[ l \right]}$进行均值和方差的归一化处理，$W^{\left[ l \right]}$和$b^{\left[ l \right]}$更加稳定，使得原来的模型也有不错的表现，提高了神经网络的复用性。

Batch Norm还有轻微的正则化（regularization）效果。具体表现在：

- 每个mini-batch都进行均值为0，方差为1的归一化操作。
- 每个mini-batch中，对各个隐藏层的$Z^{\left[ l \right]}$添加了随机噪声，效果类似于Dropout。
- mini-batch越小，正则化效果越明显。

但是，Batch Norm的正则化效果比较微弱，正则化也不是Batch Norm的主要功能。

## 3.7 测试时的Batch Norm

测试时，我们每次用一个测试样本来计算误差，对求其均值和方差是没有意义的，这就需要对$\mu $和$\sigma ^{2}$进行估计。

估计的方法有很多，理论上我们可以将所有训练集放入最终的神经网络模型中，然后将每个隐藏层计算得到的$\mu ^{\left[ l \right]}$和$\left( \sigma ^{2} \right)^{\left[ l \right]}$直接作为测试过程的$\mu $和$\sigma ^{2}$来使用。但是，实际应用中一般不使用这种方法，而是使用我们之前介绍过的指数加权平均（exponentially weighted average）的方法来预测测试过程单个样本的$\mu $和$\sigma ^{2}$。

**指数加权平均的做法是**：训练时, 记录每一个 mini-batch 在每一层的均值和方差，如$\mu ^{\left[ l \right]}$和$\left( \sigma ^{2} \right)^{\left[ l \right]}$，再计算$\mu $与$\sigma ^{2}$各自的指数加权平均，作为测试用的$\mu $与$\sigma ^{2}$。至于$\gamma $和$\beta$, 如前所述, 是同 W, b 一样的参数, 直接用训练习得的即可。

## 3.8 Softmax 回归

目前我们介绍的都是二分类问题，神经网络输出层只有一个神经元，表示预测输出$\hat{y}$是正类的概率$P\left( y=1|x \right)$，$\hat{y} > 0.5$则判断为正类，$\hat{y} < 0.5$则判断为负类。

对于多分类问题，用C表示种类个数，神经网络中输出层就有C个神经元，即$n^{\left[ L \right]}=\mbox{C}$。其中，每个神经元的输出依次对应属于该类的概率，即$P\left( y=c|x \right)$。为了处理多分类问题，我们一般使用Softmax回归模型。**Softmax回归模型输出层的激活函数**如下所示：

$z^{\left[ L \right]}=W^{\left[ L \right]}a^{\left[ L-1 \right]}+b^{\left[ L \right]}$

$a_{i}^{\left[ L \right]}=\frac{e^{z_{i}^{\left[ L \right]}}}{\sum_{i=1}^{C}{e^{z_{i}^{\left[ L \right]}}}}$

输出层每个神经元的输出$a_{i}^{\left[ L \right]}$对应属于该类的概率，满足：

$\sum_{i=1}^{C}{a_{i}^{\left[ L \right]}}=1$

其中所有的$a_{i}^{\left[ L \right]}$，即$\hat{y}$的维度为(C, 1)。

## 3.9 训练一个 Softmax 分类器

Softmax classifier的训练过程与我们之前介绍的二元分类问题有所不同。先来看一下softmax classifier的loss function。举例来说，假如C=4，某个样本的预测输出和真实输出分别为

$\hat y=\left[ \begin{array}{c} 0.3 \\ 0.2 \\ 0.1 \\ 0.4 \end{array} \right]$；$y=\left[ \begin{array}{c} 0 \\ 1 \\ 0 \\ 0 \end{array} \right]$

从$\hat{y}$值来看，$P\left( y=4|x \right)=0.4$，概率最大，而真实样本属于第2类，因此该预测效果不佳。我们定义softmax classifier的loss function为：

$L\left( \hat{y},y \right)=-\sum_{j=1}^{4}{y_{j}\cdot \log \hat{y}}$

然而，由于只有当j=2时，$y_{2}=1$，其它情况下，$y_{j}=0$。所以，上式可以简化为：

$L\left( \hat{y},y \right)=-y_{2}\cdot \log \hat{y}_{2}=-\log \hat{y}_{2}$

要让$L\left( \hat{y},y \right)$更小，就应该让$\hat{y}_{2}$越大越好。$\hat{y}_{2}$反映的是概率，完全符合我们之前的定义。

所有m个样本的cost function为：

$J=\frac{1}{m}\sum_{i=1}^{m}{L\left( \hat{y},y \right)}$

其预测输出向量$A^{\left[ L \right]}$即$\hat{Y}$的维度为(4, m)。

softmax classifier的反向传播过程仍然使用梯度下降算法，其推导过程与二元分类有一点点不一样。因为只有输出层的激活函数不一样，我们先推导$dZ^{\left[ L \right]}$：

$da^{[L]}=-\frac{1}{a^{[L]}}$

$\frac{\partial a^{[L]}}{\partial z^{[L]}}=\frac{\partial}{\partial z^{[L]}}\cdot (\frac{e^{z^{[L]}_i}}{\sum_{i=1}^Ce^{z^{[L]}_i}})=a^{[L]}\cdot (1-a^{[L]})$

$dz^{[L]}=da^{[L]}\cdot \frac{\partial a^{[L]}}{\partial z^{[L]}}=a^{[L]}-1=a^{[L]}-y$

对于所有m个训练样本：

$dZ^{[L]}=A^{[L]}-Y$

可见$dZ^{[L]}$的表达式与二元分类结果是一致的，虽然推导过程不太一样。然后就可以继续进行反向传播过程的梯度下降算法了，推导过程与二元分类神经网络完全一致。

## 3.10 深度学习框架

深度学习框架有很多，例如：Caffe/Caffe2、CNTK、DL4J、Keras、Lasagne、mxnet、PaddlePaddle、TensorFlow、Theano、Torch。

一般选择深度学习框架的基本准则是：

- 易于编程 (包括开发和部署)
- 运行速度快
- 真开放 (开源，且拥有良好的管理。那些目前开源，但缺乏管理，未来可能被闭源的不算真开放)

## 3.11 TensorFlow

TensorFlow的最大优点就是采用数据流图（data flow graphs）来进行数值运算。图中的节点（Nodes）表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组，即张量（tensor）。而且它灵活的架构让你可以在多种平台上展开计算，例如台式计算机中的一个或多个CPU（或GPU），服务器，移动设备等等。