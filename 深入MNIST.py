import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# 加载MNIST数据
mnist = read_data_sets('MNIST_data',one_hot=True)
# 运行TensorFlow的InteractiveSession
sess = tf.InteractiveSession()

#
# 构建Softmax回归模型
# 占位符
x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])

# 变量
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 初始化变量为0
# sess.run(tf.initialize_all_tables())

# 类别预测与损失函数
# y = tf.nn.softmax(tf.matmul(x,W) + b)


#
# 构建一个多层卷积神经网络
#

# 权重初始化
# 为了创建这个模型，我们需要创建大量的权重和偏置项。
# 这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度
# 由于我们使用的时ReLU神经元，因此比较好的做法时用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题（dead neurons）
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 卷积和池化
# TensorFlow在卷积和池化上有很强的灵活性
# 我们怎样处理边界？步长应该设置为多大？
# 在这个实例中，我们会一直使用vanilla版本
# 我们的卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小
# 我们的池化用简单传统的2x2大小的模板做max pooling
def conv2d(x,W):
    return  tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 第一层卷积
# 现在我们可以开始实现第一层了，它是由一个卷积连接一个max pooling完成
# 卷积在每个5x5的patch中算出32个特征，卷积的权重张量形状时[5,5,1,32],
# 前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目，而对于每一个输出通道都有一个对应的偏置量
W_conv_1 = weight_variable([5,5,1,32])
b_conv_1 = bias_variable([32])

# 为了使用这一层，我们把x变成一个4d向量，其第2,3维对应图片的宽、高，最后一维代表图片的颜色通道数
# 因为灰度图所以这里的通道数维1，如果是RGB彩图，则为3
x_image = tf.reshape(x,[-1,28,28,1])

# 我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling
h_conv_1 = tf.nn.relu(conv2d(x_image,W_conv_1) + b_conv_1)
h_pool_1 = max_pool_2x2(h_conv_1)

# 第二层卷积
# 为了构建一个更深的网络，我们会把几个类似的层堆叠起来
# 第二层中，每个5x5的patch都会得到64个特征
W_conv_2 = weight_variable([5,5,32,64])
b_conv_2 = bias_variable([64])

h_conv_2 = tf.nn.relu(conv2d(h_pool_1,W_conv_2) + b_conv_2)
h_pool_2 = max_pool_2x2(h_conv_2)

# 密集连接层
# 现在图片尺寸减到了7x7，我们有一个1024个神经元的全连接层，用于处理整个图片。
# 我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU
W_fc_1 = weight_variable([7 * 7 * 64,1024])
b_fc_1 = bias_variable([1024])

h_pool_2_flat = tf.reshape(h_pool_2,[-1,7 * 7 * 64])
h_fc_1 = tf.nn.relu(tf.matmul(h_pool_2_flat,W_fc_1) + b_fc_1)

# Dropout
# 为了减少过拟合，我们在输出层之前加入dropout，我们使用一个占位符placeholder来代表一个神经元的输出在dropout中保持不变的概率
# 这样我们可以在训练过程中启用dropout ，在测试过程中关闭dropout
# TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元的输出值scale
keep_prob = tf.placeholder("float")
h_fc_1_drop = tf.nn.dropout(h_fc_1,keep_prob)

# 输出层
# 最后，我们添加一个softmax层，就像前面的单层softmax regression一样
W_fc_2 = weight_variable([1024,10])
b_fc_2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc_1_drop,W_fc_2) + b_fc_2)

# 训练和评估模型
cross_entropy = - tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
sess.run(tf.global_variables_initializer())
for i in range(200):
    batch = mnist.train.next_batch(50)
    if i%100 == 0 :
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d, training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))


