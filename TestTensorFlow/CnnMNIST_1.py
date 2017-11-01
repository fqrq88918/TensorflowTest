import  tensorflow as tf
import  tflearn
from  tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=true)
sess = tf.InteractiveSession() # 默认的session

#测试tflearn

#定义权重，加上噪声打破完全对称，比如截断的正太分布噪声，标准差为0.1
def weight_function(shape):
    initial = tf.truncated_normal(shape,stddev=0.1) #正太分布
    return  tf.Variable(initial)

#定义偏置 因为要使用Relu，也增加0.1的正值来避免死亡节点
def bias_function(shape):
    initial = tf.constant(0.1,shape=shape)
    return  tf.Variable(initial)

#定义卷积函数 步长为1->strides = [1,1,1,1]
def conv2d_function(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding= 'SAME')

#定义池化函数 使用2x2最大池化，就是将2x2的像素块降为1x1像素; 布长为2，缩小图片尺寸->strides=[1,2,2,1]
def pool_function(x):
    return  tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


#定义输入的placeholder
x = tf.placeholder(dtype= tf.float32,shape=[0,784])
y_true = tf.placeholder(dtype= tf.float32,shape=[0,10])
#把1维输入向量转变为2维的图像，从1x784转成28x28，-1表示样本数量不固定，1表示黑白图片，1个颜色通道
xImage = tf.reshape(x,shape=[-1,28,28,1])

#定义第一个卷积层 5x5尺寸，1个颜色通道，32深度
W_conv1 = weight_function(shape=[5,5,1,32])
b_conv1 = bias_function(shape= [32])
h_conv1 = tf.nn.relu(conv2d_function(xImage,W_conv1)+b_conv1)
h_pool1 = pool_function(h_conv1)

#定义第二个卷积层 5x5尺寸，上一步的32个通道，64深度
W_conv2 = weight_function(shape=[5,5,32,64])
b_conv2 = bias_function(shape=[64])
h_conv2 = tf.nn.relu(conv2d_function(h_pool1,W_conv2)+b_conv2)
h_pool2 = pool_function(h_conv2)

#定义一个全链接层 图片尺寸经过2次2个步长的池化层缩小，已经变成28x28的4分之1，即7x7，深度为32x32
W_fc1 = weight_function(shape=[7*7*64,32*32])
b_fc1 = bias_function(shape=[1024])
#把二维转一维
h_pool2_float = tf.reshape(h_pool2,shape=[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_float,W_fc1) + b_fc1)

#定义一个dropOut层，丢弃一部分节点数据防止过拟合
keepDrop = tf.placeholder(tf.float32)
dropOut_f1 = tf.nn.dropout(h_fc1,keep_prob=keepDrop)

#定义一个softMax层,得到输出
W_fc2 = weight_function(shape=[1024,10])
b_fc2 = bias_function(10)
y_conv = tf.nn.softmax(tf.matmul(dropOut_f1,W_fc2) + b_fc2)
