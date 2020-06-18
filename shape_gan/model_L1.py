from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import scipy.misc

from ops import *
from utils import *
import h5py

class pix2pix(object):
    def __init__(self, sess, image_size=256,
                 batch_size=64, z_dim = 50, sample_size=64, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=50,
                 input_c_dim=3, output_c_dim=3, dataset_name='facades', layer_features = False,
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.layer_features = layer_features

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        if dataset_name == 'shape2im':
            self.input_c_dim = 1
            self.output_c_dim = 3
            #self.is_grayscale = True
        if self.layer_features:
            print("use layer features")

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.real_B = self.real_data[:, :, :, :self.output_c_dim]
        self.real_A = self.real_data[:, :, :, self.output_c_dim:self.input_c_dim + self.output_c_dim]
        # A is sketch, B is image
        # output_c_dim is 3, input_c_dim is 1

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        self.fake_B = self.generator(self.real_A, self.z)
        #self.real_AB = tf.concat(3, [self.real_A, self.real_B])
        #self.fake_AB = tf.concat(3, [self.real_A, self.fake_B])
        self.real_AB = tf.concat_v2([self.real_A, self.real_B],3)
        self.fake_AB = tf.concat_v2([self.real_A, self.fake_B],3)

        self.D, self.D_logits, self.h3, self.h2, self.h1, self.h0 = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_, self.h3_, self.h2_, self.h1_, self.h0_ = self.discriminator(self.fake_AB, reuse=True)
        self.fake_B_sample = self.sampler(self.real_A, self.z)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, targets=tf.ones_like(self.D)))

        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, targets=tf.zeros_like(self.D_)))

        if self.layer_features:
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, targets=tf.ones_like(self.D_))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B)) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.h3 - self.h3_)) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.h2 - self.h2_)) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.h1 - self.h1_)) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.h0 - self.h0_))
        else:
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, targets=tf.ones_like(self.D_))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))
        

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()



    def sample_model_shape2im(self, sample_images, sample_z, sample_dir, epoch, idx):
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images, self.z: sample_z}
        )
        print(samples.shape)
        save_images(samples, [8, 8],
                    './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    

    def train(self, args):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum,
            self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")


        for epoch in xrange(args.epoch):
            if self.dataset_name == 'shape2im':
                load_n = epoch % 10 + 1
                sp_name = "../code/good_txt/train_test/divid/image_shape%02d.mat" % load_n
                print(sp_name)
                f = h5py.File(sp_name,'r')
                data_X = (np.transpose(f['image_shape'])/127.5 - 1.)
                batch_idxs = min(len(data_X), args.train_size) // self.batch_size
                print(len(data_X))
                permed = np.random.permutation(len(data_X))
                data_X = data_X[permed,:]
                sample_images = data_X[0:self.batch_size]
                #scipy.misc.imsave('sketch.jpg', np.squeeze(sample_images[2,:,:,3:4]))
                #scipy.misc.imsave('image.jpg', np.squeeze(sample_images[2,:,:,0:3]))
                #raw_input("write image")
            else:
            	pass

            #print(batch_idxs)
            for idx in xrange(0, batch_idxs):
            	if self.dataset_name == 'shape2im':
                    batch = data_X[idx*self.batch_size:(idx+1)*self.batch_size]
            	else:
            	    pass
                
                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                    

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={ self.real_data: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images, self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                errG = self.g_loss.eval({self.real_data: batch_images, self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 500) == 1:
                    if self.dataset_name == 'shape2im':
                        self.sample_model_shape2im(sample_images,sample_z, args.sample_dir, epoch, idx)
                    else:
                        pass

                if np.mod(counter, 100) == 2:
                    self.save(args.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:
            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            #raw_input("press")
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4, h3, h2, h1, h0

    def generator(self, image, z, y=None):
        with tf.variable_scope("generator") as scope:

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            zb = tf.reshape(z, [self.batch_size, int(1), int(1), self.z_dim])

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            #d1 = tf.concat(3, [d1, e7])
            d1 = tf.concat_v2([d1, e7], 3)
            d1 = conv_cond_concat(d1, zb)
            #raw_input("press")
            
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat_v2([d2, e6], 3)
            d2 = conv_cond_concat(d2, zb)
            #d2 = tf.concat(3, [d2, e6])
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat_v2([d3, e5], 3)
            d3 = conv_cond_concat(d3, zb)
            #d3 = tf.concat(3, [d3, e5])
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat_v2([d4, e4], 3)
            d4 = conv_cond_concat(d4, zb)
            #d4 = tf.concat(3, [d4, e4])
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat_v2([d5, e3], 3)
            d5 = conv_cond_concat(d5, zb)
            #d5 = tf.concat(3, [d5, e3])
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat_v2([d6, e2], 3)
            d6 = conv_cond_concat(d6, zb)
            #d6 = tf.concat(3, [d6, e2])
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat_v2([d7, e1], 3)
            d7 = conv_cond_concat(d7, zb)
            #d7 = tf.concat(3, [d7, e1])
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def sampler(self, image, z, y=None):

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            zb = tf.reshape(z, [self.batch_size, int(1), int(1), self.z_dim])

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            #d1 = tf.concat(3, [d1, e7])
            d1 = tf.concat_v2([d1, e7], 3)
            d1 = conv_cond_concat(d1, zb)
            #raw_input("press")
            
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat_v2([d2, e6], 3)
            d2 = conv_cond_concat(d2, zb)
            #d2 = tf.concat(3, [d2, e6])
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat_v2([d3, e5], 3)
            d3 = conv_cond_concat(d3, zb)
            #d3 = tf.concat(3, [d3, e5])
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat_v2([d4, e4], 3)
            d4 = conv_cond_concat(d4, zb)
            #d4 = tf.concat(3, [d4, e4])
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat_v2([d5, e3], 3)
            d5 = conv_cond_concat(d5, zb)
            #d5 = tf.concat(3, [d5, e3])
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat_v2([d6, e2], 3)
            d6 = conv_cond_concat(d6, zb)
            #d6 = tf.concat(3, [d6, e2])
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat_v2([d7, e1], 3)
            d7 = conv_cond_concat(d7, zb)
            #d7 = tf.concat(3, [d7, e1])
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def test(self, args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for j in xrange(7000):
            sample_files = glob('./test/gen_openbw/gen_openbw/{:04d}/*.jpg'.format(j + 1))
            sample_image = np.array([(scipy.misc.imread(filename).astype(np.float)/127.5 - 1.) for filename in sample_files])
            sample_signP = np.array([1 for filename in sample_files])
            pad_len = self.batch_size - (len(sample_files) % self.batch_size)
            pad_img = np.array([sample_image[-1] for i in xrange(pad_len)])
            pad_fil = [sample_files[-1] for i in xrange(pad_len)]
            smaple_signN = np.array([0 for i in xrange(pad_len)])
            sample_image = np.expand_dims(np.concatenate((sample_image, pad_img), axis = 0), axis = 3)
            sample_sign = np.concatenate((sample_signP, smaple_signN), axis = 0)
            smaple_files = sample_files + pad_fil

            #smaple_img2 = np.array([sample_image[-1] for i in xrange(self.batch_size)])
            ori_z = np.ones((self.batch_size, 1)) * np.random.uniform(-1, 1, size=(1, self.z_dim))
            #added_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))
            #added_z = np.cumsum(added_z, axis=0)
            #z = ori_z + added_z
            z = ori_z

        
            print(sample_image.shape)

            batch_idxs = len(sample_image) // self.batch_size
            for idx in xrange(0, batch_idxs):
                batch_input = sample_image[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_sign = sample_sign[idx*self.batch_size:(idx+1)*self.batch_size]
                sample_outputs = self.sess.run(self.fake_B_sample, feed_dict={self.real_A: batch_input, self.z: z})
                #self.fake_B_sample_withoutD = self.sampler_withoutD(self.real_A)
                print(sample_outputs.shape)
                sample_outputs = inverse_transform(sample_outputs)

                for ind in xrange(0, self.batch_size):
                    if batch_sign[ind] == 0:
                        break;
                    directory = './test/gen_openbw/im/{:04d}/'.format(j + 1)
                    try:
                        os.stat(directory)
                    except:
                        os.mkdir(directory) 
                    out_filename = directory + os.path.basename(smaple_files[ind])
                    print(out_filename)
                    raw_input("stop")
                    scipy.misc.imsave(out_filename, np.squeeze(sample_outputs[ind,:,:,:]))
                #raw_input("here")
