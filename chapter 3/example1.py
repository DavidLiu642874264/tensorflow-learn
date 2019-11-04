__author__ = 'david.liu'
# -*- coding:UTF-8 -*-
import  tensorflow as tf

#from numpy.random import RandomState
# Numpy 是一个科学计算的工具包，这里通过Numpy 工具包生成模拟数据集。
import numpy as np

#定义训练数据batch的大小
batch_size=8

#定义神经网络的参数，这里还是沿用3.4.2小节中给出的神经网络结构
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1)) #两行三列
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1)) #三行一列

# 在shape的一个维度上使用None可以方便使用不同的batch大小。在训练时需要把数据分
# 成比较小的batch,但是在测试时，可以一次性使用全部的数据。当数据集比较小时这样比较
# 方便测试，但是数据集比较大时，将大量数据放入一个batch可能会导致内存溢出。
x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_= tf.placeholder(tf.float32,shape=(None,1),name='y-input')

#定义神经网络的前向传播的过程。
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

# 定义损失函数和反向传播的算法
y=tf.sigmoid(y)
cross_entropy=-tf.reduce_mean(
    y_*tf.log(tf.clip_by_value(1-y,1e-10,1.0))
    +(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
train_step = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集。
rdm=np.random.RandomState(1)
dataset_size=128
X=rdm.rand(dataset_size,2)
# 定义规则来给出样本的标签。在这里所有x1+x2<1的样例都被认为时正样本（比如零件合格），
# 而其他为负样本（比如零件不合格）。和Tensorflow游乐场中的表示法不大一样的地方是，
# 在这里使用 0 来表示负样本，1表示正样本。大部分解决分类问题的神经网络都会采用0和1的表示方法。
Y=[[int(x1+x2<1)] for(x1,x2) in X]

# 创建一个会话来运行Tensorflow程序
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    #初始化变量
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))
    '''
    训练之前神经网络的参数的值：
    w1=[[-0.8113182   1.4845988   0.06532937] [-2.4427042   0.0992484   0.5912243 ]]
    w2=[[-0.8113182 ] [ 1.4845988 ] [ 0.06532937]]
    '''

    #设定训练的轮数。
    STEPS =5000
    for i in range(STEPS):
        # 每次选取batch_size  个样本进行训练。
        start=(i* batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)
        # 通过选取的样本训练神经网络并更新参数。
        sess.run(train_step,
                 feed_dict={x:X[start:end],y_:Y[start:end]}  )
        if i%1000==0:
            #每隔一段时间计算在所有数据上的交叉熵并输出。
            total_cross_entropy=sess.run(
                cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training step(s),cross_entropy on all data is %g" %
                      (i,total_cross_entropy))
            '''
            输出结果：
            After 0 training step(s),cross_entropy on all data is 2.6202
            After 1000 training step(s),cross_entropy on all data is 2.61707
            After 2000 training step(s),cross_entropy on all data is 2.61313
            After 3000 training step(s),cross_entropy on all data is 2.60872
            After 4000 training step(s),cross_entropy on all data is 2.60401

            通过这个结果可以发现随着训练的进行，交叉熵是逐渐变小的。
            交叉熵越小说明预测的结果和真是的结果差距越小。
            '''
    print(sess.run(w1))
    print(sess.run(w2))
    '''
    在训练之后神经网络的参数的值：
    w1=[[-0.80726564  1.4805404   0.06133108] [-2.4382334   0.0947869   0.58683354]]
    w2=[[-0.8066256 ] [ 1.4804041 ] [ 0.06077295]]
    可以发现这两个参数的取值以将发生了变化，这个变化就是训练的结果。
    它使得这个神经网络能更好的拟合提供的训练数据。
    '''






