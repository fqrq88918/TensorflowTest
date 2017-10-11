import ssl
import  tensorflow as tf

ssl._create_default_https_context = ssl._create_unverified_context
from  tensorflow.examples.tutorials.mnist import  input_data
_mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)


_sess = tf.InteractiveSession() #将_sess注册成默认的Session
_x = tf.placeholder(tf.float32,[None,784]) #创建1个输入数据，None表示不限条数，784表示784维度的向量
_w = tf.Variable(tf.zeros([784,10])) #创建权重 784维度，10分类
_bias = tf.Variable(tf.zeros([10]))#创建偏执，10分类
_y = tf.nn.softmax(tf.matmul(_x,_w)+_bias) #sofrmax算法，tf.matmul是矩阵乘法函数,预测的概率分布

trueY = tf.placeholder(tf.float32,[None,10])#创建真实概率分布的输入，10分类，既label
cross_entropy = tf.reduce_mean(-tf.reduce_sum(trueY*tf.log(_y),reduction_indices=[1])) # 信息熵
#tf.reduce_mean 求平均值 #reduce_sum 求和  降低维度

trainStep  = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #调用用于训练的优化算法，学习率0。5
#优化目标是信息熵
_init =  tf.global_variables_initializer()#全局初始化
_sess.run(_init)

#迭代训练 随机抽取100条样本
for i in range(1000):
 batch_X,batch_Y  =  _mnist.train.next_batch(100)
 trainStep.run({_x:batch_X,trueY : batch_Y})
correct = tf.equal(tf.argmax(_y,1),tf.argmax(trueY,1)) #比较概率
accurary = tf.reduce_mean(tf.cast(correct,tf.float32))#压缩
print(accurary.eval({_x:_mnist.test.images,trueY:_mnist.test.labels}))
