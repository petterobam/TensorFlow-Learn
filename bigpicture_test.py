# -*- coding: utf-8 -*-

# 计算图概念说明
# 下面是 graph , session , operation , tensor 四个概念的简介。
# Tensor：类型化的多维数组，图的边；
# Operation:执行计算的单元，图的节点；
# Graph：一张有边与点的图，其表示了需要进行计算的任务；
# Session:称之为会话的上下文，用于执行图。

# mnist识别实例
# 1、为输入X与输出y定义placeholder；
# 2、定义权重W；
# 3、定义模型结构；
# 4、定义损失函数；
# 5、定义优化算法。


import tensorflow as tf

# 首先，需要下载数据集，mnist数据可以在 http://yann.lecun.com/exdb/mnist/ 下载到，也可以通过如下两行代码得到。
import tensorflow.examples.tutorials.mnist.input_data as input_data
# 下载数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 首先导入需要的包，定义 X 与 y 的 placeholder 以及 W,b 的 Variables
# None表示任意维度，一般是min-batch的 batch size。而 W 定义是 shape 为784,10，rank为2的Variable，b是shape为10，rank为1的Variable。
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 定义模型
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 求逻辑回归的损失函数，这里使用了 cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
# 这里的 cross entropy 取了均值。定义了学习步长为0.5，使用了梯度下降算法（GradientDescentOptimizer）最小化损失函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 不要忘记初始化 Variables。
init = tf.global_variables_initializer()
# 最后，我们的 graph 至此定义完毕，下面就可以进行真正的计算，包括初始化变量，输入数据，并计算损失函数与利用优化算法更新参数
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        # 迭代了1000次，每次输入了100个样本。mnist.train.next_batch 就是生成下一个 batch 的数据，这里知道它在干什么就可以。
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # 这里使用单纯的正确率，正确率是用取最大值索引是否相等的方式，因为正确的 label 最大值为1，而预测的 label 最大值为最大概率。
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels}))
# 至此，我们开发了一个简单的手写数字识别模型。

# 总结全文，我们首先介绍了 graph 与 session，并解释了基础数据结构，
# 讲解了一些Variable需要注意的地方并介绍了 placeholders 与 feed_dict 。
# 最终以一个手写数字识别的实例将这些点串起来，希望可以给想要入门的你一丢丢的帮助。