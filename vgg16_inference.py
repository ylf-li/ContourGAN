import numpy as np
import tensorflow as tf
import tensorlayer as tl
# from utils import *
from tensorlayer.layers import *

class vgg16:
    def __init__(self, imgs):
        self.imgs = imgs
        self.convlayers()
        self.deconde()

    def convlayers(self):
        self.parameters = []

        with tf.name_scope('encoder1') as scope:
            # zero-mean input
            with tf.name_scope('preprocess') as scope:
                mean = tf.constant([122.6789, 116.779, 104.0069], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
                images = self.imgs-mean

            # conv1_1
            with tf.name_scope('conv1_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv1_2
            with tf.name_scope('conv1_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool1

            self.pool1 = tf.nn.max_pool(self.conv1_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')

            # conv2_1
            with tf.name_scope('conv2_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv2_2
            with tf.name_scope('conv2_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool2

            self.pool2 = tf.nn.max_pool(self.conv2_2,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool2')

            # conv3_1
            with tf.name_scope('conv3_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv3_2
            with tf.name_scope('conv3_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv3_3
            with tf.name_scope('conv3_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool3

            self.pool3 = tf.nn.max_pool(self.conv3_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool3')

            # conv4_1
            with tf.name_scope('conv4_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv4_2
            with tf.name_scope('conv4_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv4_3
            with tf.name_scope('conv4_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool4
            self.pool4 = tf.nn.max_pool(self.conv4_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 1,1, 1],
                                   padding='SAME',
                                   name='pool4')

        with tf.name_scope('encoder2') as scope:
            # conv5_1
            with tf.name_scope('conv5_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1), name='weights')
                conv = tf.nn.atrous_conv2d(self.pool4, kernel, 2, padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv5_2
            with tf.name_scope('conv5_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.atrous_conv2d(self.conv5_1, kernel, 2, padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv5_3
            with tf.name_scope('conv5_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1), name='weights')
                conv = tf.nn.atrous_conv2d(self.conv5_2, kernel, 2, padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]


    def deconde(self):

        with tf.variable_scope("decoder"):

            batch_size = tf.shape(self.conv5_3)[0]
            w_init = tf.contrib.layers.variance_scaling_initializer()

            shape=self.conv4_1.get_shape().as_list()
            self.conv5_1.set_shape(shape)
            self.conv5_2.set_shape(shape)
            self.conv5_3.set_shape(shape)

            conv5_1 = InputLayer(self.conv5_1,name='g/h1/conv5_1')
            conv5_1 = Conv2d(conv5_1, 32, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h1/conv51')
            conv5_2 = InputLayer(self.conv5_2,name='g/h1/conv5_2')
            conv5_2 = Conv2d(conv5_2, 32, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h1/conv52')
            conv5_3 = InputLayer(self.conv5_3,name='g/h1/conv5_3')
            conv5_3 = Conv2d(conv5_3, 32, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h1/conv53')
            conv4_1 = InputLayer(self.conv4_1,name='g/h2/conv4_1')
            conv4_1 = Conv2d(conv4_1, 32, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h2/conv41')
            conv4_2 = InputLayer(self.conv4_2,name='g/h2/conv4_2')
            conv4_2 = Conv2d(conv4_2, 32, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h2/conv42')
            conv4_3 = InputLayer(self.conv4_3,name='g/h2/conv4_3')
            conv4_3 = Conv2d(conv4_3, 32, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h2/conv43')
            concat=ConcatLayer([conv5_3,conv5_2,conv5_2,conv4_3,conv4_2,conv4_1],concat_dim=3,name='g/h2/concat')
            concat=Conv2d(concat, 256, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='h/h22/concat')
            self.deconv1=UpSampling2dLayer(concat,size=(2,2),is_scale=True,method=0,name ='upsample2')
            self.deconv1=Conv2d(self.deconv1, 256, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='h/h2/conv3')


            conv3_1 = InputLayer(self.conv3_1,name='g/h3/conv3_1')
            conv3_1 = Conv2d(conv3_1, 32, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h3/conv31')
            conv3_2 = InputLayer(self.conv3_2,name='g/h3/conv3_2')
            conv3_2 = Conv2d(conv3_2, 32, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h3/conv32')
            conv3_3 = InputLayer(self.conv3_3,name='g/h3/conv3_3')
            conv3_3 = Conv2d(conv3_3, 32, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h3/conv33')
            concat=ConcatLayer([self.deconv1,conv3_3,conv3_1,conv3_2],concat_dim=3,name='g/h3/concat')
            concat=Conv2d(concat, 128, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='h/h33/concat')
            self.deconv2=UpSampling2dLayer(concat,size=(2,2),is_scale=True,method=0,name ='upsample2h3')
            self.deconv2=Conv2d(self.deconv2, 128, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='h/h3/conv3')


            conv2_1 = InputLayer(self.conv2_1,name='g/h4/conv2_1')
            conv2_1 = Conv2d(conv2_1, 16, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h4/conv22')
            conv2_2 = InputLayer(self.conv2_2,name='g/h4/conv2_2')
            conv2_2 = Conv2d(conv2_2, 16, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h4/conv32')
            concat=ConcatLayer([self.deconv2,conv2_2,conv2_1],concat_dim=3,name='g/h4/concat')
            concat=Conv2d(concat, 64, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='h/h44/concat4')
            self.deconv3=UpSampling2dLayer(concat,size=(2,2),is_scale=True,method=0,name ='upsample2h4')
            self.deconv3=Conv2d(self.deconv3, 64, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='h/h4/conv4')
            

            conv1_1 = InputLayer(self.conv1_1,name='g/h5/conv1_1')
            conv1_1 = Conv2d(conv1_1, 16, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h5/conv11')
            conv1_2 = InputLayer(self.conv1_2,name='g/h5/conv1_2')
            conv1_2 = Conv2d(conv1_2, 16, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='g/h5/conv12')
            concat= ConcatLayer([self.deconv3,conv1_2,conv1_1],concat_dim=3,name='g/h5/concat')
            concat=Conv2d(concat, 16, (3,3), (1, 1),act=tf.nn.relu,padding='SAME', W_init=w_init, name='h/h55/conv5')

            self.pred=Conv2d(concat, 1, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/pre/conv')




def discriminator(inputs, is_train=True, reuse=False):

    c_dim = 3
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='d/in')
        net_h0 = Conv2d(net_in, 3, (1,1), (1,1), act=tf.nn.relu,padding='SAME', W_init=w_init, name='d/h0/conv2d0')
        net_h01 = Conv2d(net_h0, 32, (5, 5), (2, 2), act=tf.nn.relu,padding='SAME', W_init=w_init, name='d/h0/conv2d01')
        pool1 = MaxPool2d(net_h01, (2, 2), padding='SAME', name='pool1')

        net_h1 = Conv2d(pool1, 64, (5, 5), (2, 2), act=tf.nn.relu,padding='SAME', W_init=w_init, name='d/h1/conv2d1')
        net_h12 = Conv2d(net_h1, 64, (5, 5), (2, 2), act=tf.nn.relu,padding='SAME', W_init=w_init, name='d/h1/conv2d12')
        pool2 = MaxPool2d(net_h12, (2, 2), padding='SAME', name='pool2')

        net_h21 = Conv2d(pool2, 64, (5, 5), (2, 2), act=tf.nn.relu,padding='SAME', W_init=w_init, name='d/h2/conv2d21')
        net_h22 = Conv2d(net_h21, 64, (5, 5), (2, 2), act=tf.nn.relu,padding='SAME', W_init=w_init, name='d/h2/conv2d22')
        pool3 = MaxPool2d(net_h22, (2, 2), padding='SAME', name='pool3')

        net_h31 = Conv2d(pool3, 64, (5, 5), (2, 2), act=tf.nn.relu,padding='SAME', W_init=w_init, name='d/h2/conv2d31')
        net_h32 = Conv2d(net_h31, 64, (5, 5), (2, 2), act=tf.nn.relu,padding='SAME', W_init=w_init, name='d/h2/conv2d32')
        pool4 = MaxPool2d(net_h32, (2, 2), padding='SAME', name='pool4')

        net_h4 = FlattenLayer(pool4, name='d/h4/flatten')
        net_h5 = DenseLayer(net_h4, n_units=1000, act=tf.tanh,W_init = w_init, name='d/h5/lin_sigmoid')
        net_h6 = DenseLayer(net_h5, n_units=100, act=tf.tanh,W_init = w_init, name='d/h6/lin_sigmoid')
        net_h7 = DenseLayer(net_h6, n_units=1, act=tf.identity,W_init = w_init, name='d/h7/lin_sigmoid')	

        logits = net_h7.outputs
        net_h7.outputs = tf.nn.sigmoid(net_h7.outputs)
    return net_h7, logits

