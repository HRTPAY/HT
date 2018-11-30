#
# 准备数据
# MNIST是机器学习领域的一个经典问题，指的是让机器查看一系列大小为28x28像素的手写数字灰度图像，并判断这些图像代表0-9中的哪个数字
#

#
# 下载
# 在run_training()方法的一开始，
# input_data.read_data_sets()函数会确保你的本地训练文件夹中，已经下载了正确的数据，
# 然后将这些数据解压并返回一个含有DataSet实例的字典
# data_sets = input_data.read_data_sets(FLAGS.train_dir,FLAGS.fake_data)
# 注意，fake_data标记是用于单元测试的，读者可以不必理会

# 数据集                   目的
# data_sets.train           55000个图像和标签（labels)，作为主要训练集
# data_sets.validation      5000个图像和标签，用于迭代验证训练准确度
# date_sets.test            10000个图像和标签，用于最终测试训练准确度(train accuracy)

# 输入和占位符（Inputs and Placeholders)
# placeholder_inputs()函数将生成两个tf.placeholder操作，定义传入图表中的shape参数，
# shape参数中包括batch_size值，后续还会将实际的训练用例传入图表
# 在训练循环的后续步骤中，传入的整个图像和标签数据集会被切片，以符合每一个操作所设置的batch_size值，
# 占位符操作将会填补以符合这个batch_size值，然后使用feed_dict参数，将数据传入sess.run()函数

#
# 构建图表(Build the Graph)
# 在为数据创建占位符后，就可以运行mnist.py文件，经过三阶段的模式函数操作：
# inference(),loss()和training().图表就构建完成
# inference(),尽可能的构建好图表，满足促使神经网络向前反馈并做出预测的要求
# loss(),往inference图表中添加生成损失(loss)所需要的操作(ops)
# training()，往损失图表中添加计算并应用梯度(gradients)所需要的操作

#
# 推理（Inference)
# inference()函数会尽可能地构建图表，做到返回包含了预测结果（output prediction）的Tensor
# 它接受图像占位符为输入，在此基础上借助ReLU（Rectified Linear Units)激活函数，构建一对完全连接层（layers)
# 以及一个有着十个节点(node),指明了输出logits模型的线性层
# 每一层都创建于一个唯一的tf.name_scope之下，创建于该作用域之下的所有元素都将带有其前缀
# with tf.name_scope('hidden_1') as scope:
# 在定义的作用域中，每一层所使用的权重和偏差都在tf.Variable实例中生成，并且包含了各自期望的shape
# weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS,hidden_1_units],stddev=1.0/math.sqrt(float(IMAGE_PIXELS))),name='weights')
# biases = tf.Variable(tf.zero([hidden_1_units]),name='biases')
# 例如，当这些层是在hidden_1作用域下生成时，赋予权重变量的独特名称就是"hidden_1/weights"
# 每个变量在构建时，都会获得初始化操作（initializer ops)
# 在这种最常见的情况下，通过tf.truncated_normal函数初始化权重变量
# 给赋予的shape是一个二维tensor，其中第一个维度代表该层中权重变量所连接(connect from)的单元数量，第二个维度代表该层中权重变量所连接到的单元数量
# 对于一个名为hidden_1的第一层，相应的维度则是[IMAGE_PIXELS,hidden_1_units],因为权重变量将图像输入连接到了hidden_1层
# tf.truncated_normal初始函数将根据所得到的均值和标准差，生成一个随机分布
# 然后，通过tf.zeros函数初始化偏差变量(biases)，确保所有偏差的起始值都是0，而它们的shape则是其在该层中所连接的单元数量
# 图表的三个主要操作，分别是两个tf.nn.relu操作，它们中嵌入隐藏层所需的tf.matmul，以及logits模型所需要的另外一个tf.matmul
# 三者依次生成，各自的tf.Variable实例则与输入占位符或下一层的输出tensor所连接

#
# 损失(Loss)
# loss()函数通过添加所需要的损失函数，进一步构建图表
# 首先，labels_placeholder中的值，将被编码为一个含有1-hot values的Tensor。
# 例如，如果类标识符为'3'，那么该值就会被转换为[0,0,0,1,0,0,0,0,0,0]
# batch_size = tf.size(labels)
# labels = tf.expand_dims(labels,1)
# indices = tf.expand_dims(tf.range(0,batch_size,1),1)
# concated = tf.concat(1,[indices,labels])
# onehot_labels = tf.sparse_to_dense(concated,tf.pack([batch_size,NUM_CLASSES]),1.0,0.0)
# 之后，又添加一个tf.nn.softmax_cross_entropy_with_logits(logits,onehot_labels,name='xentropy')
# 然后，使用tf.reduce_mean函数，计算batch维度（第一维度）下交叉熵（cross entropy)的平均值，将该值作为总损失
# loss = tf.reduce_mean(cross_entropy,name='xentropy_mean')
# 最后，程序会返回包含了损失值得Tensor

#
# 训练
# training()函数添加了通过梯度下降（gradient descent）将损失最小化所需的操作
# 首先，该函数从loss()函数中获取损失Tensor，
# 将其交给tf.scalar_summary,后者与SummaryWriter配合使用，可以向事件文件（events file)中生成汇总值
# tf.scalar_summary(loss.op.name,loss)
# 接下来，我们实例化一个tf.train.GradientDesentOptimizer,负责按照所要求的学习效率(learning rate)应用梯度下降法
# optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
# 之后，我们生成一个变量用于保存全局训练步骤（global trainning step)的数值，并使用minimize()函数更新系统中三角权重（triangle weights),
# 增加全局步骤的操作。根据惯例，这个操作被称为train_op,是TensorFlow会话（session)诱发一个完整训练步骤所必须运行的操作

#
# 图表
# 在run_training()这个函数的一开始，是一个python语言中的with命令，这个命令表明所有已构建的操作都要与默认的tf.Graph全局实例关联
# tf.Graph()实例是一系列可以作为整体执行的操作，TensorFlow的大部分应用场景只需要默认图表一个实例即可

#
# 会话
# 完成全部的构建准备，生成全部所需的操作后，我们创建一个tf.Session,用于运行图表
# sess = tf.Session()
# 另外，也可以利用with代码块生成Session,限制作用域
# with tf.Session() as sess:
# Session函数没有传入参数，代表该代码将会依附于默认的本地会话（如果还没有创建会话，则会创建新的会话），
# 生成会话后，所有tf.Variable实例都会立即通过调用各自初始化操作中的sess.run()函数进行初始化
# init = tf.initialize_all_variables()
# sess.run(init)
# sess.run()方法将会运行图表中与作为参数传入的操作相对应的完整子集。
# 在初次调用，init操作只包含了变量初始化程序tf.group

#
# 训练循环
# 完成会话中变量的初始化后，就可以开始训练了
# 训练的每一步都是通过用户代码控制完成，而能实现有效训练的最简单循环就是
# for step in range(max_steps):
#     session.run(train_op)

# 向图表提供反馈
# 执行每一步时，我们的代码会生成一个反馈字典（feed dictionary),其中包含对应步骤中训练所使用的例子，这些例子中的哈希值就是其所代表的占位符操作
# fill_feed_dict函数会给查询指定的DataSet，索要下一批次batch_size的图像和标签，与占位符相匹配的Tensor则会包含下一批次的图像和标签
# images_feed,labels_feed = data_set.next_batch(FLAGS.batch_size)
# 然后，以占位符为哈希值，创建一个Python字典对象，键值则是其代表的反馈Tensor
# feed_dict={
#     images_placeholder:images_feed,
#     labels_placeholder:labels_feed,
# }

# 检查状态
# 在运行sess.run函数时，要在代码中明确其需要获取的两个值：[train_op,loss]
# for step in range(FLAGS.max_steps):
#       feed_dict = fill_feed_dict(date.sets.train,images_placeholder,labels_placeholder)
#       loss_value = sess.run([train_op,loss],feed_dict=feed_dict)
# 因为要获取这两个值，sess.run()会返回一个有两个元素的元组。其中每一个Tensor对象，对应了返回的元组中的numpy数组，
# 而这些数组中包含了当前这步训练中对应的Tensor的值。由于train_op并不会产生输出，在其返回的元组中对应的元素就是None，所有被抛弃
# 但是如果在训练中出现偏差，loss Tensor的值可能就会变成NaN，所以我们要获取它的值，并记录
# 假设训练一切正常，没有出现NaN，训练循环就会每隔100个训练步骤，就打印一行简单的状态文本，告知当前训练状态
# if step % 100 == 0:
#   print("Step %d:loss = % .2f(%.3f sec)"%(step,loss_value,duration))

# 状态可视化
# 为了释放TensorBoard所使用的事件文件（events file),所有的即时数据都要在图表构建阶段合并到一个操作中
# summary_op = tf.merge_all_summaries()
# 在创建好会话后，可以实例化一个tf.train.SummaryWriter,用于写入包含了图表本身和即时数据具体值的事件文件
# summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,graph_def=sess.graph_def)
# 最后，每次运行summary_op时，都会往事件文件中写入最新的即时数据，函数的输出会传入事件文件读写器（writer)的add_summary()函数
# summary_str = sess.run(summary,op,feed_dict=feed_dict)
# summary_writer.add_summary(summary_str,step)
# 事件文件写入完毕后，可以就训练文件夹打开一个TensorBoard，查看即时数据的情况


