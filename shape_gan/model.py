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
import scipy.io as sio

class au2shape(object):
	def __init__(self, sess,
				 batch_size=1, sample_size=1, gf_dim=64, df_dim=64, L1_lambda=10, shape_size = 34, feature_size = 17, time_frame = 100, num_category = 20, num_cont = 3, dataset_name='facades', layer_features = False,
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
		self.batch_size = batch_size
		self.sample_size = sample_size
		self.layer_features = layer_features
		self.shape_size = shape_size
		self.feature_size = feature_size * time_frame

		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.L1_lambda = L1_lambda
		self.num_category = num_category
		self.num_cont = num_cont

		
		if self.layer_features:
			print("use layer features")

		# batch normalization : deals with poor initialization helps gradient flow
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')
		self.d_bns = batch_norm(name='d_bns')

		self.g_bn_e2 = batch_norm(name='g_bn_e2')
		self.g_bn_e3 = batch_norm(name='g_bn_e3')
		self.g_bn_e4 = batch_norm(name='g_bn_e4')


		self.g_bn_d1 = batch_norm(name='g_bn_d1')
		self.g_bn_d2 = batch_norm(name='g_bn_d2')
		self.g_bn_d3 = batch_norm(name='g_bn_d3')
		self.g_bn_d4 = batch_norm(name='g_bn_d4')
		self.g_bn_d5 = batch_norm(name='g_bn_d5')


		self.dataset_name = dataset_name
		self.checkpoint_dir = checkpoint_dir
		self.build_model()

	def build_model(self):
		# input is A (temporal AUs)
		self.real_input = tf.placeholder(tf.float32,
										[self.batch_size, self.feature_size],
										name='real_input')
		
		self.real_ids = tf.placeholder(tf.float32,
										[self.batch_size, 1], name = 'real_ids')
		self.auxi_ids = tf.placeholder(tf.float32,
										[self.batch_size, 1], name = 'auxi_ids')

		# output is B (shape parameters)
		self.real_output = tf.placeholder(tf.float32,
										[self.batch_size, self.shape_size],
										name='real_output')

		self.auxi_output = tf.placeholder(tf.float32,
										[self.batch_size, self.shape_size],
										name='auxi_output')

		
		self.z_cont = tf.placeholder(tf.float32, [self.batch_size, self.num_cont], name='z_cont') # continuous z
		self.z_ca_r = tf.one_hot(tf.cast(tf.squeeze(self.real_ids, -1), tf.int32), depth = self.num_category) # category constraint for real input
		self.z_ca_a = tf.one_hot(tf.cast(tf.squeeze(self.auxi_ids, -1), tf.int32), depth = self.num_category) # category constraint for auxiliary input
		self.z_r = tf.concat_v2([self.z_cont, self.z_ca_r], 1) # concatenate on dimension 1 (dim 0 is for batch use)
		self.z_a = tf.concat_v2([self.z_cont, self.z_ca_a], 1) # concatenate on dimension 1 (dim 0 is for batch use)

		self.fake_B_r = self.generator(self.real_input, self.z_r, reuse=False) #generate using real input and related category data
		self.fake_B_a = self.generator(self.real_input, self.z_a, reuse=True) # generate using real input but different category data. reuse
		

		self.D_r, self.D_logits_r, self.h2_r, self.h1_r, self.h0_r, self.cat_logits_r, _  = self.discriminator(self.real_output, reuse=False)
		self.D_a, self.D_logits_a, self.h2_a, self.h1_a, self.h0_a, self.cat_logits_a, _  = self.discriminator(self.auxi_output, reuse=True)
		self.D__r, self.D_logits__r, self.h2__r, self.h1__r, self.h0__r, self.cat_logits__r, self.cont__r = self.discriminator(self.fake_B_r, reuse=True)
		self.D__a, self.D_logits__a, self.h2__a, self.h1__a, self.h0__a, self.cat_logits__a, self.cont__a = self.discriminator(self.fake_B_a, reuse=True)
		self.fake_B_sample_r = self.sampler(self.real_input, self.z_r)
		self.fake_B_sample_a = self.sampler(self.real_input, self.z_a)

		self.d_sum = tf.summary.histogram("d", self.D_r + self.D_a)
		self.d__sum = tf.summary.histogram("d_", self.D__r + self.D__a)
		#self.fake_B_sum = tf.summary.image("fake_B", self.fake_B_r)

		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_r, targets=tf.ones_like(self.D_r))) \
							+ tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_a, targets=tf.ones_like(self.D_a)))

		self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits__r, targets=tf.zeros_like(self.D__r))) \
							+ tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits__a, targets=tf.zeros_like(self.D__a)))

		# generation loss for real and auxi data
		self.g_loss_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits__r, targets=tf.ones_like(self.D__r))) \
						+ self.L1_lambda * tf.reduce_mean(tf.abs(self.real_output - self.fake_B_r)) \
						+ self.L1_lambda * tf.reduce_mean(tf.abs(self.h2_r - self.h2__r)) \
						+ self.L1_lambda * tf.reduce_mean(tf.abs(self.h1_r - self.h1__r)) \
						+ self.L1_lambda * tf.reduce_mean(tf.abs(self.h0_r - self.h0__r))

		self.g_loss_a = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits__a, targets=tf.ones_like(self.D__a))) \
						+ 0.2 * self.L1_lambda * tf.reduce_mean(tf.abs(self.h2_a - self.h2__a)) 
						#+ self.L1_lambda * tf.reduce_mean(tf.abs(self.h1_a - self.h1__a)) \
						#+ self.L1_lambda * tf.reduce_mean(tf.abs(self.h0_a - self.h0__a))

		
		# category loss for real, auxi, fake_r and fake_a
		self.cat_loss_real_r = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cat_logits_r, labels=self.z_ca_r))
		self.cat_loss_real_a = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cat_logits_a, labels=self.z_ca_a))
		self.cat_loss_fake_r = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cat_logits__r, labels=self.z_ca_r))
		self.cat_loss_fake_a = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cat_logits__a, labels=self.z_ca_a))
		self.cat_loss = self.cat_loss_real_r + self.cat_loss_real_a + self.cat_loss_fake_r + self.cat_loss_fake_a

		# continuous loss for real and auxi
		self.con_loss_r = tf.reduce_mean(tf.square(self.cont__r-self.z_cont))
		self.con_loss_a = tf.reduce_mean(tf.square(self.cont__a-self.z_cont))
		self.con_loss = self.con_loss_r + self.con_loss_a


		self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

		self.d_loss = self.d_loss_real + self.d_loss_fake
		self.g_loss = self.g_loss_r + self.g_loss_a

		self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
		self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		self.saver = tf.train.Saver()

	

	def train(self, args):
		"""Train pix2pix"""
		d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
						  .minimize(self.d_loss + self.cat_loss + self.con_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
						  .minimize(self.g_loss + self.cat_loss + self.con_loss, var_list=self.g_vars)

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)

		self.g_sum = tf.summary.merge([self.d__sum, self.d_loss_fake_sum, self.g_loss_sum])
		self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
		self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

		counter = 1
		start_time = time.time()

		if self.load(self.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")


		au_name = "../../code/shape_train/all_tr_auo.mat"
		f = h5py.File(au_name,'r')
		real_input = (np.transpose(f['all_tr_auo']))
		real_input = np.reshape(real_input, [len(real_input), -1]) 

		pv_name = "../../code/shape_train/all_tr_pu.mat"
		matcontent = sio.loadmat(pv_name)
		real_output = np.array(matcontent['all_tr_pu'])
		real_output = real_output[:, 6:40]

		id_name = "../../code/shape_train/all_user_ids.mat"
		matcontent = sio.loadmat(id_name)
		id_X = np.array(matcontent['all_user_ids'])

		id_emotion_name = "../../code/shape_train/all_emotion_ids.mat"
		matcontent = sio.loadmat(id_emotion_name)
		id_emotion = np.array(matcontent['all_emotion_ids'])

		
		print(real_input.shape)
		print(id_X.shape)
		raw_input("stop here")

		batch_idxs = min(len(real_input), args.train_size) // self.batch_size

		

		id_A = np.copy(id_X)
		output_A = np.copy(real_output)

		for epoch in xrange(args.epoch):
			if self.dataset_name == 'shapes':
				# if we have multi files (e.g. 10) in an epoch, use the following:
				#load_n = epoch % 10 + 1
				#print(load_n)
				#sp_name = "../code/good_txt/train_test/divid/image_shapec%02d.mat" % load_n
				#f = h5py.File(sp_name,'r')
				#data_X = (np.transpose(f['image_shapec'])/127.5 - 1.)
				#batch_idxs = min(len(data_X), args.train_size) // self.batch_size
				#print(len(data_X))
				#permed = np.random.permutation(len(data_X))
				#data_X = data_X[permed,:]
				#sample_images = data_X[0:self.batch_size]

				# if we have only one file in an epoch, use the following:

				permed = np.random.permutation(len(real_input))
				real_input = real_input[permed,:]
				real_output = real_output[permed,:]
				sample_input = (real_input[0:self.batch_size])
				id_X = id_X[permed]
				id_emotion = id_emotion[permed]
				sample_ids = id_X[0:self.batch_size]
				sample_z_cont = np.random.normal(0, 1, size=(self.batch_size , self.num_cont))


				for i in xrange(0, len(id_X)):
					while 1:
						j = np.random.randint(0, len(id_X))
						if id_X[j] != id_X[i] and id_emotion[j] == id_emotion[i]:
							id_A[i] = id_X[j]
							output_A[i] = real_output[j]
							break

				sample_ids_a = id_A[0:self.batch_size]

				#print(id_X[0])
				#print(id_A[0])

				#raw_input("stop here")

			else:
				pass

			#print(batch_idxs)
			for idx in xrange(0, batch_idxs):
				if self.dataset_name == 'shapes':
					batch_input = np.array(real_input[idx*self.batch_size:(idx+1)*self.batch_size]).astype(np.float32)
					batch_output = np.array(real_output[idx*self.batch_size:(idx+1)*self.batch_size]).astype(np.float32)
					ids = id_X[idx*self.batch_size:(idx+1)*self.batch_size]
					batch_a = np.array(output_A[idx*self.batch_size:(idx+1)*self.batch_size]).astype(np.float32)
					ids_a = id_A[idx*self.batch_size:(idx+1)*self.batch_size]
					z_cont = np.random.normal(0, 1, size=(self.batch_size , self.num_cont))
				else:
					pass
				#raw_input("stop here")
				# Update D network
				_, summary_str = self.sess.run([d_optim, self.d_sum],
											   feed_dict={ self.real_input: batch_input,
											   			   self.real_output: batch_output,
														   self.auxi_output: batch_a,
														   self.real_ids: ids,
														   self.auxi_ids: ids_a,
														   self.z_cont: z_cont})
				self.writer.add_summary(summary_str, counter)

				# Update G network
				_, summary_str = self.sess.run([g_optim, self.g_sum],
											   feed_dict={ self.real_input: batch_input,
											   			   self.real_output: batch_output,
														   self.auxi_output: batch_a,
														   self.real_ids: ids,
														   self.auxi_ids: ids_a,
														   self.z_cont: z_cont})
				self.writer.add_summary(summary_str, counter)

				# Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
				_, summary_str = self.sess.run([g_optim, self.g_sum],
											   feed_dict={ self.real_input: batch_input,
											   			   self.real_output: batch_output,
														   self.auxi_output: batch_a,
														   self.real_ids: ids,
														   self.auxi_ids: ids_a,
														   self.z_cont: z_cont})
				self.writer.add_summary(summary_str, counter)

				errD_fake = self.d_loss_fake.eval({ self.real_input: batch_input,
											   		self.real_output: batch_output,
													self.auxi_output: batch_a,
													self.real_ids: ids,
													self.auxi_ids: ids_a,
													self.z_cont: z_cont})
				errD_real = self.d_loss_real.eval({ self.real_input: batch_input,
											   		self.real_output: batch_output,
													self.auxi_output: batch_a,
													self.real_ids: ids,
													self.auxi_ids: ids_a,
													self.z_cont: z_cont})
				errG = self.g_loss.eval({ self.real_input: batch_input,
										  self.real_output: batch_output,
										  self.auxi_output: batch_a,
										  self.real_ids: ids,
										  self.auxi_ids: ids_a,
										  self.z_cont: z_cont})

				counter += 1
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
					% (epoch, idx, batch_idxs,
						time.time() - start_time, errD_fake+errD_real, errG))

				#if np.mod(counter, 200) == 1:
				#	if self.dataset_name == 'shape2im':
				#		self.sample_model_shape2im(sample_images,sample_ids, sample_ids_a, sample_z_cont, args.sample_dir, epoch, idx)
				#	else:
				#		pass

				if np.mod(counter, 2000) == 2:
					self.save(args.checkpoint_dir, counter)

	def discriminator(self, shape, y=None, reuse=False):

		with tf.variable_scope("discriminator") as scope:
			# image is 256 x 256 x (input_c_dim + output_c_dim)
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse == False

			h0 = lrelu(linear(shape, self.df_dim*2, 'd_h0_lin'))
			h1 = lrelu(self.d_bn1(linear(h0, self.df_dim*4, 'd_h1_lin')))
			# h1 is (64 x 4)
			h2 = lrelu(self.d_bn2(linear(h1, self.df_dim*4, 'd_h2_lin')))
			# h2 is (64 x 8)
			h3 = linear(h2, 1, 'd_h3_lin')
			h_share = lrelu(self.d_bns(linear(h2, 128, 'd_hs_lin')))

			# recog_cat is used for recognizing categories
			recog_cat = linear(h_share, self.num_category, 'd_cat_lin')
			# recog_con is used for recognizing continous variables
			recog_con = linear(h_share, self.num_cont, 'd_con_lin')

			return tf.nn.sigmoid(h3), h3, h2, h1, h0, recog_cat, tf.nn.sigmoid(recog_con)

	def generator(self, feature, z, y=None, reuse=False):
		with tf.variable_scope("generator") as scope:

			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse == False

			
			#zb = tf.reshape(z, [self.batch_size, int(1), int(1), self.num_category + self.num_cont])
			#z is okay, dont have to resize it
			# image is (256 x 256 x input_c_dim)
			e1 = linear(feature, self.gf_dim*2, 'g_e1_lin')
			e1 = tf.concat_v2([e1, z], 1)
			# e1 is (128 x 128 x self.gf_dim)
			e2 = self.g_bn_e2(linear(lrelu(e1), self.gf_dim*2, 'g_e2_lin'))
			e2 = tf.concat_v2([e2, z], 1)
			# e2 is (64 x 64 x self.gf_dim*2 + )
			e3 = self.g_bn_e3(linear(lrelu(e2), self.gf_dim*2, 'g_e3_lin'))
			e3 = tf.concat_v2([e3, z], 1)
			# e3 is (32 x 32 x self.gf_dim*4 + )
			e4 = self.g_bn_e4(linear(lrelu(e3), self.gf_dim*2, 'g_e4_lin'))
			e4 = tf.concat_v2([e4, z], 1)
			# e4 is (16 x 16 x self.gf_dim*8 + )
			


			d1 = linear(tf.nn.relu(e4),self.gf_dim*2, 'g_d1')
			d1 = tf.nn.dropout(self.g_bn_d1(d1), 0.5)
			d1 = tf.concat_v2([d1, e4], 1)
			#raw_input("press")
			

			d2 = linear(tf.nn.relu(d1),self.gf_dim*2, 'g_d2')
			d2 = tf.nn.dropout(self.g_bn_d2(d2), 0.5)
			d2 = tf.concat_v2([d2, e3], 1)
			

			d3 = linear(tf.nn.relu(d2), self.gf_dim*2, 'g_d3')
			d3 = tf.nn.dropout(self.g_bn_d3(d3), 0.5)
			d3 = tf.concat_v2([d3, e2], 1)
			

			d4 = linear(tf.nn.relu(d3),self.gf_dim*2, 'g_d4')
			d4 = self.g_bn_d4(d4)
			d4 = tf.concat_v2([d4, e1], 1)
			
			

			self.d5 = linear(tf.nn.relu(d4),self.shape_size, 'g_d5')

			return self.d5

	def sampler(self, feature, z, y=None):

		with tf.variable_scope("generator") as scope:
			scope.reuse_variables()

			e1 = linear(feature, self.gf_dim*2, 'g_e1_lin')
			e1 = tf.concat_v2([e1, z], 1)
			# e1 is (128 x 128 x self.gf_dim)
			e2 = self.g_bn_e2(linear(lrelu(e1), self.gf_dim*2, 'g_e2_lin'))
			e2 = tf.concat_v2([e2, z], 1)
			# e2 is (64 x 64 x self.gf_dim*2 + )
			e3 = self.g_bn_e3(linear(lrelu(e2), self.gf_dim*2, 'g_e3_lin'))
			e3 = tf.concat_v2([e3, z], 1)
			# e3 is (32 x 32 x self.gf_dim*4 + )
			e4 = self.g_bn_e4(linear(lrelu(e3), self.gf_dim*2, 'g_e4_lin'))
			e4 = tf.concat_v2([e4, z], 1)
			# e4 is (16 x 16 x self.gf_dim*8 + )
			


			d1 = linear(tf.nn.relu(e4),self.gf_dim*2, 'g_d1')
			d1 = tf.nn.dropout(self.g_bn_d1(d1), 0.5)
			d1 = tf.concat_v2([d1, e4], 1)
			#raw_input("press")
			

			d2 = linear(tf.nn.relu(d1),self.gf_dim*2, 'g_d2')
			d2 = tf.nn.dropout(self.g_bn_d2(d2), 0.5)
			d2 = tf.concat_v2([d2, e3], 1)
			

			d3 = linear(tf.nn.relu(d2), self.gf_dim*2, 'g_d3')
			d3 = tf.nn.dropout(self.g_bn_d3(d3), 0.5)
			d3 = tf.concat_v2([d3, e2], 1)
			

			d4 = linear(tf.nn.relu(d3),self.gf_dim*2, 'g_d4')
			d4 = self.g_bn_d4(d4)
			d4 = tf.concat_v2([d4, e1], 1)
			
			

			self.d5 = linear(tf.nn.relu(d4),self.shape_size, 'g_d5')
			return self.d5


	def save(self, checkpoint_dir, step):
		model_name = "au2shape.model"
		model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoint...")

		model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False


	def test_gen(self, args):
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)

		if self.load(self.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		onset_name = "../../code/onSet.mat"
		matcontent = sio.loadmat(onset_name)
		onSet = np.array(matcontent['onSet'])

		te_first_ids_name = "../../code/te_first_ids.mat"
		matcontent = sio.loadmat(te_first_ids_name)
		te_first_ids = np.array(matcontent['te_first_ids'])

		user_ids_name = "../../code/user_ids.mat"
		matcontent = sio.loadmat(user_ids_name)
		user_ids = np.array(matcontent['user_ids'])

		for j in xrange(1):  # should be len(onSet)

			temp_name = "../../selected_video_audio/{:d}/mat/frm_no_clips.mat".format(onSet[j, 0].astype(int))
			matcontent = sio.loadmat(temp_name)
			total_len = len(np.array(matcontent['frm_no_clips']))
			cur_id = user_ids[onSet[j, 0].astype(int) - 1]

			for k in xrange(te_first_ids[j], total_len + 1):
				auo_name = "../../selected_video_audio/{:d}/mat/auo{:03d}.mat".format(onSet[j, 0].astype(int), k)
				matcontent = sio.loadmat(auo_name)
				auo = np.array(matcontent['auo'])
				au_input = np.squeeze([auo[100+p:200+p, :] for p in xrange(len(auo) - 199)])
				au_input = np.reshape(au_input, [len(au_input), -1]) 
				old_len = len(au_input)
				pad_len = self.batch_size - (old_len % self.batch_size)
				pad_auo = np.array([au_input[-1] for i in xrange(pad_len)])
				au_input = np.concatenate((au_input, pad_auo), axis = 0)

				cur_ids = [cur_id for p in xrange(self.batch_size)]
				z_cont = [0, 0, 0]
				pad_zs = [z_cont for i in xrange(self.batch_size)]
				batch_idxs = min(len(au_input), args.train_size) // self.batch_size

				directory = '../../selected_video_audio/{:d}/mat/test/'.format(onSet[j, 0].astype(int))
				try:
					os.stat(directory)
				except:
					os.mkdir(directory)

				sample_outputs = np.array([]).reshape(0,self.shape_size).astype('float32')


				for idx in xrange(0, batch_idxs):
					batch_input = au_input[idx*self.batch_size:(idx+1)*self.batch_size]
					sample_output = self.sess.run(self.fake_B_sample_r, feed_dict={self.real_input: batch_input,self.real_ids: cur_ids,self.z_cont: pad_zs})
					sample_outputs = np.vstack((sample_outputs,sample_output))

				sample_outputs = sample_outputs[0:old_len]
				out_filename = directory + 'pvut{:03d}.mat'.format(k)
				sio.savemat(out_filename, {'pvut':sample_outputs})
					
