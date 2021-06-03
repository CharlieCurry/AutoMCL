# -*- coding:utf8 -*
import tensorflow as tf
import numpy as np
import sys
import time

def build_layer5_net(batch_size, l1_size, l2_size, l3_size, l4_size, out_size):
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    xs = tf.placeholder(tf.float32, [None, l1_size], name='xs')
    ys = tf.placeholder(tf.float32, [None, out_size], name='ys')
    sess = tf.Session()
    # train_data = np.random.random([batch_size,l1_size])
    # test_data = np.random.random([batch_size,out_size])
    # train_x = train_data[:, :27]
    # train_y = train_data[:, 27:28]
    # test_x = test_data[:, :27]
    # test_y = test_data[:, 27:28]
    def add_layer(input, in_size, out_size, activation_function, name):
        Weight = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size])) + 0.000000001
        Wx_plus_b = tf.matmul(input, Weight, name=name) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, name=name)
        return outputs

    L1 = add_layer(xs, l1_size, l2_size, activation_function=tf.nn.relu, name='L1')
    L2 = add_layer(L1, l2_size, l3_size, activation_function=tf.nn.relu, name='L2')
    L3=add_layer(L2,l3_size,l4_size,activation_function=tf.nn.relu, name='L3')
    prediction = add_layer(L3, l4_size, out_size, activation_function=tf.nn.relu, name='prediction')

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    hubers = tf.losses.huber_loss(ys, prediction, delta=2.0)
    hubers_loss = tf.reduce_sum(hubers)
    train_step = tf.train.RMSPropOptimizer(0.0005).minimize(hubers_loss)
    init = tf.global_variables_initializer()
    sess.run(init)
    # batch_size = 512
    # data_size = len(train_x)
    # STEPS = 60001
    # for i in range(STEPS):
    #     start = (i * batch_size) % data_size
    #     end = min(start + batch_size, data_size)
    #     # sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    #     sess.run(train_step, feed_dict={xs: train_x[start:end], ys: train_y[start:end], keep_prob: 0.8})
    #     if i % 2000 == 0:
    #         print("i=", i, "train_loss=", sess.run(hubers_loss, feed_dict={xs: train_x, ys: train_y, keep_prob: 1}))
    #         print("i=", i, "valiation_loss=", sess.run(hubers_loss, feed_dict={xs: test_x, ys: test_y, keep_prob: 1}))

    saver = tf.train.Saver()
    saver.save(sess, 'net/X_2')


def build_layer7_net(batch_size, l1_size, l2_size, l3_size, l4_size, l5_size,l6_size, out_size):
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    xs = tf.placeholder(tf.float32, [None, l1_size], name='xs')
    ys = tf.placeholder(tf.float32, [None, out_size], name='ys')
    sess = tf.Session()
    # train_data = np.random.random([batch_size,l1_size])
    # test_data = np.random.random([batch_size,out_size])
    # train_x = train_data[:, :27]
    # train_y = train_data[:, 27:28]
    # test_x = test_data[:, :27]
    # test_y = test_data[:, 27:28]
    def add_layer(input, in_size, out_size, activation_function, name):
        Weight = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size])) + 0.000000001
        Wx_plus_b = tf.matmul(input, Weight, name=name) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, name=name)
        return outputs

    L1 = add_layer(xs, l1_size, l2_size, activation_function=tf.nn.relu, name='L1')
    L2 = add_layer(L1, l2_size, l3_size, activation_function=tf.nn.relu, name='L2')
    L3=add_layer(L2,l3_size,l4_size,activation_function=tf.nn.relu, name='L3')
    L4 = add_layer(L3, l4_size, l5_size, activation_function=tf.nn.relu, name='L4')
    L5 =add_layer(L4,l5_size,l6_size,activation_function=tf.nn.relu, name='L5')
    prediction = add_layer(L5, l6_size, out_size, activation_function=tf.nn.relu, name='prediction')

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    hubers = tf.losses.huber_loss(ys, prediction, delta=2.0)
    hubers_loss = tf.reduce_sum(hubers)
    train_step = tf.train.RMSPropOptimizer(0.0005).minimize(hubers_loss)
    init = tf.global_variables_initializer()
    sess.run(init)
    # batch_size = 512
    # data_size = len(train_x)
    # STEPS = 60001
    # for i in range(STEPS):
    #     start = (i * batch_size) % data_size
    #     end = min(start + batch_size, data_size)
    #     # sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    #     sess.run(train_step, feed_dict={xs: train_x[start:end], ys: train_y[start:end], keep_prob: 0.8})
    #     if i % 2000 == 0:
    #         print("i=", i, "train_loss=", sess.run(hubers_loss, feed_dict={xs: train_x, ys: train_y, keep_prob: 1}))
    #         print("i=", i, "valiation_loss=", sess.run(hubers_loss, feed_dict={xs: test_x, ys: test_y, keep_prob: 1}))

    saver = tf.train.Saver()
    saver.save(sess, 'net/X_2')


def evaluation(batch_size,l1_size,out_size):
    print("batch size=",batch_size)
    meta = "./net/X_2.meta"
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(meta)
        new_saver.restore(sess, tf.train.latest_checkpoint('./net/'))
        graph = tf.get_default_graph()
        valiation_x_1 = np.random.uniform(0,1,[batch_size,l1_size])
        valiation_y = np.random.uniform(0,1,[batch_size,out_size])
        y = graph.get_tensor_by_name("prediction:0")
        xs = graph.get_tensor_by_name("xs:0")
        ys = graph.get_tensor_by_name("ys:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        print(y)
        s1 = time.time()
        prediction_y = sess.run(y, feed_dict={xs: valiation_x_1, ys: valiation_y, keep_prob:1.0})
        e1 = time.time()
        print('time(ms):', ((e1 - s1))*1000)

if __name__ == "__main__":
    #16 128 256 512 1024 1000
    # batch_size=int(sys.argv[1])
    # l1_size=int(sys.argv[2])
    # l2_size=int(sys.argv[3])
    # l3_size=int(sys.argv[4])
    # l4_size=int(sys.argv[5])
    # out_size=int(sys.argv[6])

    batch_size, l1_size, l2_size, l3_size, l4_size, out_size = 1,128 ,256 ,512, 1024 ,1000
    l5_size, l6_size = 2048, 4096

    print("warm up...")
    build_layer5_net(batch_size, l1_size, l2_size, l3_size, l4_size, out_size)
    #build_layer7_net(batch_size, l1_size, l2_size, l3_size, l4_size,l5_size, l6_size, out_size)
    evaluation(1, l1_size, out_size)
    print("testing...")
    evaluation(1,l1_size,out_size)
    evaluation(16, l1_size, out_size)
    evaluation(32, l1_size,  out_size)
    evaluation(64, l1_size, out_size)
    evaluation(128, l1_size,  out_size)
    evaluation(256, l1_size, out_size)
    evaluation(512, l1_size,  out_size)




