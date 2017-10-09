
import glob
import os
import sys
import random
import shutil
import cv2
import numpy as np
from vgg16_inference import vgg16,discriminator
# from models import vgg16

import tensorflow as tf
import tensorflow.contrib as tc
import tensorlayer as tl
from tensorlayer.layers import *
from scipy.misc import imread, imresize,imsave
import matplotlib.pyplot as plt


batch_size=8
epochs=4

start_learning_rate=1e-5
DGAN_learning_rate=1e-5
GGAN_learning_rate=1e-5
weight_decay=0.0002

crop_size_H=400
crop_size_W=400
iter_step=0

summaries_dir='summary'
shutil.rmtree('results')
os.mkdir('results')
shutil.rmtree('summary')
os.mkdir('summary')

imglist=np.loadtxt('/opt/data/HED-BSDS/train_pair.txt',dtype='str')
# imglist=imglist[:28800]

num_batch =int(len(imglist)/batch_size)
print('loadind pretrained model')
model=np.load('model.npz')

global_steps = tf.Variable(0, trainable=False)
input_img=tf.placeholder(tf.float32,[None,crop_size_H,crop_size_W,3],name='input_image')
GT = tf.placeholder(tf.float32,[None,crop_size_H,crop_size_W],name='GT')

vgg=vgg16(input_img)
Contour=tf.squeeze(tf.sigmoid(vgg.pred.outputs))
pred=tf.squeeze(vgg.pred.outputs)

# discriminator loss
# net_d, d_logits = discriminator(tf.concat(3,[tf.sigmoid(vgg.pred.outputs),input_img]), is_train=True, reuse=False)
net_d, d_logits = discriminator(tf.sigmoid(vgg.pred.outputs), is_train=True, reuse=False)
net_d2, d2_logits = discriminator(tf.expand_dims(GT,-1), is_train=True, reuse=True)
d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')
d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
d_loss = d_loss_real + d_loss_fake
tf.summary.scalar('dis_loss', d_loss)

# vars list
vgg_vars = tl.layers.get_variables_with_name('encoder',True,True)
gen_vars = tl.layers.get_variables_with_name('decoder', True, True)
dis_vars   = tl.layers.get_variables_with_name('discriminator', True, True)

# Context loss
regularizer=tf.contrib.layers.apply_regularization(\
             tf.contrib.layers.l2_regularizer(weight_decay),vgg_vars+gen_vars)
g_gan_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')
loss_context = tf.nn.weighted_cross_entropy_with_logits(pred,GT,15)
loss_context = tf.reduce_mean(loss_context)+regularizer
g_loss = loss_context + 0.01*g_gan_loss
tf.summary.scalar('loss_context', loss_context)
learning_rate = tf.train.exponential_decay(start_learning_rate,\
                             global_steps,40000, 0.1, staircase=True)
g_optim = tf.train.AdamOptimizer(learning_rate,0.9).\
            minimize(g_loss, global_step=global_steps,var_list=vgg_vars+gen_vars)
d_optim = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(d_loss, var_list=dis_vars)


merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    train_writer = tf.summary.FileWriter('summary', sess.graph) 
    for i,x in enumerate(model['params'][:26]):
        sess.run(vgg_vars[i].assign(x))
    for i,x in enumerate(model['params'][26:]):
        sess.run(gen_vars[i].assign(x))
    # print('Model restored')
    # saver.restore(sess,"models/BSDS/models_BSDS500_weighted_SGD.ckpt")
    all_names=[]
    for epoch in xrange(epochs):
        np.random.shuffle(imglist)
        index_begin=0
        for index,batch_indx in enumerate(xrange(num_batch)):
            resall=[]
            imgbatch=imglist[index_begin:index_begin+batch_size]
            index_begin=index_begin+batch_size

            imgs_raw=[cv2.imread(x[0]) for x in imgbatch]; imgs=np.array(imgs_raw)
            gts_raw=[cv2.imread(x[1],0)/255.0 for x in imgbatch];gts=np.array(gts_raw)

            gts[gts>0.5]=1.0

            iter_step = iter_step + 1

            if(0):
                lr,g,gerarate_loss,res,summary,_ = sess.run([learning_rate,global_steps,\
                        loss_context,Contour,merged,g_optim],\
                        feed_dict={GT:gts,vgg.imgs:imgs})
                resall.extend(res)
                train_writer.add_summary(summary,iter_step)
                save=[imsave(os.path.join('results','{}.png'.format(index)),resall[i]) \
                            for i in np.arange(0,len(resall))]
                print("Epoch: [%2d/%2d] [%4d/%4d] learing_rate: %.8f global_step: %d  context_loss: %.6f" \
                    %(epoch, epochs, index, num_batch, lr,g,gerarate_loss))
                if((index+1)%500==0):
                    saver.save(sess,"models/BSDS/models_BSDS500_weighted_SGD.ckpt")

            else:

                dis_loss,_,summary = sess.run([d_loss,d_optim,merged],\
                            feed_dict={GT:gts,vgg.imgs:imgs})

                res,gerarate_loss,gen_loss,g_ganloss,summary,_= sess.run([Contour,loss_context,\
                        g_loss,g_gan_loss,merged,g_optim],feed_dict={GT:gts,vgg.imgs:imgs})

                resall.extend(res)
                train_writer.add_summary(summary,iter_step)
                save=[imsave(os.path.join('results','{}.png'.format(index)),\
                       1./(1+np.exp(-resall[i]))) for i in np.arange(0,len(resall))]  
                print("Epoch: [%2d/%2d] [%4d/%4d] d_loss: %.6f,g_loss: %.6f,gerarate_loss: %.6f" \
                    %(epoch, epochs, index, num_batch,dis_loss,gen_loss,gerarate_loss))
                if((index+1)%500==0):
                    saver.save(sess,"models/BSDS/models_BSDS500_weighted_adam_GAN.ckpt")