import tensorflow as tf
import numpy as np
# 定义测试函数
def lm_test(variable):
    init = tf.global_variables_initializer()  # 初始化所有变量
    sess = tf.Session()
    sess.run(init)
    print(sess.run(variable))

# # 定义初始化函数
# def Init():
#     init = tf.global_variables_initializer()  # 初始化所有变量
#     sess = tf.Session()
#     sess.run(init)
#     return sess
# 构建一元二次方程函数
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 构建300个点 分布在-1 到1 区间，
# 直接采用np 生成等差数列的方法，并将结果为300 个点的一维数组，转换为300×1 的二维数组
noice = np.random.normal(0, 0.5, x_data.shape)  # 加入噪声点,和x一个维度 均值0 方差0.05
y_data = np.square(x_data) - 0.5 + noice  # 构建函数 y = x ^ 2 - 0.5 + 噪声
# weights = tf.Variable(tf.random_normal([2, 20]))
# lm_test(x_data)
# 定义x和y的占位符
xs = tf.placeholder(tf.float32, [None, 1])  # 不限制维度
ys = tf.placeholder(tf.float32, [None, 1])  # 不限制维度

# 这里我们需要构建一个隐藏层和一个输出层。作为神经网络中的层，输入参数应该有4 个
# 变量：输入数据、输入数据的维度、输出数据的维度和激活函数。每一层经过向量化（y =
# weights×x + biases）的处理，并且经过激活函数的非线性化处理后，最终得到输出数据.



def addlayer(inputs, in_size, out_size, activation_fuc = None):
    # 构建权重：in_size×out_size 大小的矩阵
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 构建偏置 1 x out_size
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 矩阵乘法
    wx_b = tf.matmul(inputs, weights) + biases
    if activation_fuc is None:
        outputs = wx_b
    else:
        outputs = activation_fuc(wx_b)  # 假如有激活函数，就激活
    return outputs  # 得到输出数据

# 构建隐藏层 假设20个神经元
h1 = addlayer(xs, 1, 20, activation_fuc=tf.nn.relu)

# 构建输出层 输出层的输入为隐藏层的输出 1个神经元
prediction = addlayer(h1, 20, 1, activation_fuc=None)

# 接下来需要构建损失函数：计算输出层的预测值和真实值间的误差，对二者差的平方求和
# 再取平均，得到损失函数。运用梯度下降法，以0.1 的效率最小化损失：

# 计算预测值和真实值间的误差

# reduce_xxx这类操作，在官方文档中，都放在Reduction下：
# ReductionTensorFlow provides several operations that
# you can use to perform common math computations that reduce various dimensions of a tensor.
# 也就是说，这些操作的本质就是降维，以xxx的手段降维。在所有reduce_xxx系列操作中，都有reduction_indices这个参数，
# 即沿某个方向，使用xxx方法，对input_tensor进行降维。reduction_indices的默认值是None，即把input_tensor降到 0维，也就是一个数。
# 对于2维input_tensor，reduction_indices=0时，按列；reduction_indices=1时，按行。


loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 梯度下降

# 我们让TensorFlow 训练1000 次，每50 次输出训练的损失值：
# 初始化Session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# 循环遍历
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys:y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys:y_data}))