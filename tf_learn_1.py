# -*- coding: utf-8 -*-
import tensorflow as tf

print(tf.VERSION)
print(tf.keras.__version__)

from tensorflow.keras import layers

# 要构建一个简单的全连接网络（即多层感知器）
# 在 Keras 中，您可以通过组合层来构建模型。模型（通常）是由层构成的图。最常见的模型类型是层的堆叠：tf.keras.Sequential 模型。
print("1.序列模型")
model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))

# 我们可以使用很多 tf.keras.layers，它们具有一些相同的构造函数参数：
# activation：设置层的激活函数。此参数由内置函数的名称指定，或指定为可调用对象。默认情况下，系统不会应用任何激活函数。
# kernel_initializer 和 bias_initializer：创建层权重（核和偏差）的初始化方案。此参数是一个名称或可调用对象，默认为 "Glorot uniform" 初始化器。
# kernel_regularizer 和 bias_regularizer：应用层权重（核和偏差）的正则化方案，例如 L1 或 L2 正则化。默认情况下，系统不会应用正则化函数。
# 以下代码使用构造函数参数实例化 tf.keras.layers. Dense 层
# Create a sigmoid layer:
print("2.配置层")
layers.Dense(64, activation='sigmoid')
# Or:
layers.Dense(64, activation=tf.sigmoid)

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer='orthogonal')

# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

# 构建好模型后，通过调用 compile 方法配置该模型的学习流程：
# tf.keras.Model.compile 采用三个重要参数：
# optimizer：此对象会指定训练过程。从 tf.train 模块向其传递优化器实例，
# 例如 tf.train.AdamOptimizer、tf.train.RMSPropOptimizer 或 tf.train.GradientDescentOptimizer。
# loss：要在优化期间最小化的函数。
# 常见选择包括均方误差 (mse)、categorical_crossentropy 和 binary_crossentropy。
# 损失函数由名称或通过从 tf.keras.losses 模块传递可调用对象来指定。
# metrics：用于监控训练。它们是 tf.keras.metrics 模块中的字符串名称或可调用对象。
print("3.训练和评估")
print("3.1.设置训练流程")
model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu'),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

# 对于小型数据集，请使用内存中的 NumPy 数组训练和评估模型。使用 fit 方法使模型与训练数据“拟合”：
import numpy as np
print("3.2.输入 NumPy 数据")
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
# tf.keras.Model.fit 采用三个重要参数：
# epochs：以周期为单位进行训练。一个周期是对整个输入数据的一次迭代（以较小的批次完成迭代）。
# batch_size：当传递 NumPy 数据时，模型将数据分成较小的批次，并在训练期间迭代这些批次。此整数指定每个批次的大小。
# 请注意，如果样本总数不能被批次大小整除，则最后一个批次可能更小。
# validation_data：在对模型进行原型设计时，您需要轻松监控该模型在某些验证数据上达到的效果。
# 传递此参数（输入和标签元组）可以让该模型在每个周期结束时以推理模式显示所传递数据的损失和指标。
model.fit(data, labels, epochs=10, batch_size=32)
# 下面是使用 validation_data 的示例
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))

# 使用 Datasets API 可扩展为大型数据集或多设备训练。将 tf.data.Dataset 实例传递到 fit 方法
# Instantiates a toy dataset instance:
print("3.3.输入 tf.data 数据集")
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(dataset, epochs=10, steps_per_epoch=30)
# 在上方代码中，fit 方法使用了 steps_per_epoch 参数（表示模型在进入下一个周期之前运行的训练步数）。
# 由于 Dataset 会生成批次数据，因此该代码段不需要 batch_size。
# 数据集也可用于验证：
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)
# tf.keras.Model.evaluate 和 tf.keras.Model.predict 方法可以使用 NumPy 数据和 tf.data.Dataset。
# 要评估所提供数据的推理模式损失和指标，请运行以下代码：
print("3.4.评估和预测")
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.evaluate(data, labels, batch_size=32)

model.evaluate(dataset, steps=30)
# 要在所提供数据（采用 NumPy 数组形式）的推理中预测最后一层的输出，请运行以下代码：
result = model.predict(data, batch_size=32)
print(result.shape)