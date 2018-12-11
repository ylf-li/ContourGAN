

import os
import shutil
import cv2
import argparse
import tensorlayer as tl
from tensorlayer.layers import *
from scipy.misc import imsave
from vgg16_inference import vgg16
from tensorpack.tfutils.optimizer  import  AccumGradOptimizer

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--batch_size', default=5,
                    help='The batch size of images numbers')
parser.add_argument('--epochs', default=20,
                    help='The iteration numbers ')
parser.add_argument('--learning_rate', default=1e-6,
                    help='The start learning of the network')
parser.add_argument('--weight_decay', default=0.0002,
                    help='weight decay used in weights')
parser.add_argument('--crop_size_H', default=400,
                    help='The weight of input image')
parser.add_argument('--crop_size_W', default=400,
                    help='The weight of input image size')
parser.add_argument('--dropout', default=0.9,
                    help='The number of keep prob')

iter_step = 0
args = parser.parse_args()


imglist=np.loadtxt('train_pair.lst',dtype='str')
num_batch =int(len(imglist)/args.batch_size)


global_steps = tf.Variable(0, trainable=False)
input_img=tf.placeholder(tf.float32,[None,args.crop_size_H,args.crop_size_W,3],name='input_image')
GT = tf.placeholder(tf.float32,[None,args.crop_size_H,args.crop_size_W],name='GT')
Pos_Mask = tf.placeholder(tf.float32,[None,args.crop_size_H,args.crop_size_W],name='pos_mask')
Neg_Mask = tf.placeholder(tf.float32,[None,args.crop_size_H,args.crop_size_W],name='neg_mask')
keep_prob = tf.placeholder(tf.float32)


print('loadind model VGG')
weights = np.load('vgg16_weights.npz')
keys = sorted(weights.keys())

vgg=vgg16(input_img)


Contour=tf.squeeze(tf.nn.sigmoid(vgg.pred.outputs))
pred=tf.squeeze(vgg.pred.outputs)

vgg_vars = tl.layers.get_variables_with_name('encoder',True,True)
gen_vars = tl.layers.get_variables_with_name('decoder', True, True)

regularizer=tf.contrib.layers.apply_regularization(\
             tf.contrib.layers.l2_regularizer(args.weight_decay),vgg_vars+gen_vars)

y = tf.cast(GT, tf.float32)

count_neg = tf.reduce_sum(Neg_Mask)
count_pos = tf.reduce_sum(Pos_Mask)
beta = count_neg / (count_neg + 1.1*count_pos)

pos_weight = beta / (1 - beta)
cost = tf.nn.weighted_cross_entropy_with_logits(logits=pred, targets=y, pos_weight=pos_weight)
cost = tf.reduce_mean(cost * (1 - beta)*(Pos_Mask + Neg_Mask))
zero = tf.equal(count_pos, 0.0)
loss_context = tf.where(zero, 0.0, cost)

loss_context = tf.reduce_mean(loss_context)+regularizer
tf.summary.scalar('loss_context', loss_context)
learning_rate = tf.train.exponential_decay(args.learning_rate,global_steps,40000, 0.1, staircase=True)
AdamOpt = tf.train.AdamOptimizer(learning_rate,0.9)
# like caffe iter_size
AccumGradOpt = AccumGradOptimizer(AdamOpt,2)
opt = AccumGradOpt.minimize(loss_context,var_list=vgg_vars+gen_vars)

merged = tf.summary.merge_all()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True

with tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    train_writer = tf.summary.FileWriter('summary_drop', sess.graph)
    for i,k in enumerate(keys[:25]):
         sess.run(vgg.parameters[i].assign(weights[k]))
    # checkpoint = tf.train.get_checkpoint_state("models/BSDS/")
    # if checkpoint and checkpoint.model_checkpoint_path:
    #     saver.restore(sess, checkpoint.model_checkpoint_path)
    #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
    # else:
    #     print("Could not find old network weights")

    for epoch in xrange(args.epochs):
        np.random.shuffle(imglist)
        index_begin=0
        for index,batch_indx in enumerate(xrange(num_batch)):
            resall=[]
            imgbatch=imglist[index_begin:index_begin+args.batch_size]
            index_begin=index_begin+args.batch_size

            imgs_raw=[cv2.resize(cv2.imread(x[0]),(400,400)) for x in imgbatch];
            imgs=np.array(imgs_raw)
            gts_raw=[cv2.resize(cv2.imread(x[1],0)/255.0,(400,400)) for x in imgbatch];
            gts=np.array(gts_raw)

            gts[gts > 0.5] = 1.0

            pos_mask = np.zeros((args.batch_size,400,400))
            neg_mask = np.zeros((args.batch_size,400,400))
            pos_mask[gts>0.5] = 1.0
            neg_mask[gts==0.0] = 1.0


            iter_step = iter_step + 1

            posw,lr,g,gerarate_loss,res,summary,_ = sess.run([pos_weight,learning_rate,global_steps,\
                    loss_context,Contour,merged,opt],\
                    feed_dict={GT:gts,vgg.imgs:imgs,
                               Pos_Mask:pos_mask,Neg_Mask:neg_mask,
                               keep_prob:args.dropout})
            resall.extend(res)
            train_writer.add_summary(summary,iter_step)
            save=[imsave(os.path.join('results','{}.png'.format(index)),resall[i]) \
                        for i in np.arange(0,len(resall))]
            print("Epoch: [%2d/%2d] [%4d/%4d] learing_rate: %.8f pos_weight:%.4f context_loss: %.6f" \
                %(epoch, args.epochs, index, num_batch, lr,posw,gerarate_loss))

        saver.save(sess,"models/BSDS/models_BSDS500_epoch_{}.ckpt".format(epoch))