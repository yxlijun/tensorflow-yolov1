from __future__ import division 
from __future__ import print_function 
from __future__ import absolute_import 

import tensorflow as tf 
import numpy as np 

class Vgg19(object):
	"""docstring for Vgg16"""
	def __init__(self,vgg19_npy_path=None,trainable=True):
		super(Vgg19, self).__init__()
		if vgg19_npy_path is not None:
		    self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
		else:
		    self.data_dict = None

		self.trainable = trainable

		
	def forward(self,images):
		self.conv1_1 = self.conv_layer(images, 3, 64, "conv1_1")
		self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
		self.pool1 = self.max_pool(self.conv1_2, 'pool1')

		self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
		self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
		self.pool2 = self.max_pool(self.conv2_2, 'pool2')

		self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
		self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
		self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
		self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
		self.pool3 = self.max_pool(self.conv3_4, 'pool3')

		self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
		self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
		self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
		self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
		self.pool4 = self.max_pool(self.conv4_4, 'pool4')

		self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
		self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
		self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
		self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
		self.pool5 = self.max_pool(self.conv5_4, 'pool5')

		self.conv6_1 = self.det_conv_layer(self.pool5,512,256,"conv6_1")
		self.conv7_1 = self.det_conv_layer(self.conv6_1,256,256,"conv7_1")
		self.conv8_1 = self.det_conv_layer(self.conv7_1,256,256,"conv8_1")

		self.conv_end = self.det_conv_layer(self.conv8_1,256,30,'conv9_1',relu=False)

		sigmoid_out = tf.sigmoid(self.conv_end,name='output')
		return sigmoid_out


	def det_conv_layer(self,bottom,in_channels,out_channels,name,relu=True):
		with tf.variable_scope(name):
			residual = tf.identity(bottom)
			filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name+"_1")
			conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
			bias = tf.nn.bias_add(conv, conv_biases)
			bias = tf.layers.batch_normalization(bias,training=self.trainable,name=name+'_bn1')

			filt2,conv_biases2 = self.get_conv_var(3,out_channels,out_channels,name+"_2")
			conv2 = tf.nn.conv2d(bias,filt2,[1,1,1,1],padding='SAME')
			bias2 = tf.nn.bias_add(conv2,conv_biases2)
			bias2 = tf.layers.batch_normalization(bias2,training=self.trainable,name=name+'_bn2')

			if in_channels!=out_channels:
				filt3,conv_biases3 = self.get_conv_var(3,in_channels,out_channels,name+'_3')
				conv3 = tf.nn.conv2d(residual,filt3,[1,1,1,1],padding='SAME')
				bias3 = tf.nn.bias_add(conv3,conv_biases3)
				residual = tf.layers.batch_normalization(bias3,training=self.trainable,name=name+'_bn3')

			out = tf.nn.relu(residual+bias2) if relu else (residual+bias2)

			return out


	def max_pool(self, bottom, name):
	    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


	def conv_layer(self, bottom, in_channels, out_channels, name):
	    with tf.variable_scope(name):
	        filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

	        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
	        bias = tf.nn.bias_add(conv, conv_biases)
	        relu = tf.nn.relu(bias)

	        return relu

	def get_conv_var(self, filter_size, in_channels, out_channels, name):
	    initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
	    filters = self.get_var(initial_value, name, 0, name + "_filters")

	    initial_value = tf.truncated_normal([out_channels], .0, .001)
	    biases = self.get_var(initial_value, name, 1, name + "_biases")

	    return filters, biases

	def get_var(self, initial_value, name, idx, var_name):
		if self.data_dict is not None and name in self.data_dict:
		    value = self.data_dict[name][idx]
		else:
		    value = initial_value

		var = tf.Variable(value, name=var_name)

		# print var_name, var.get_shape().as_list()
		assert var.get_shape() == initial_value.get_shape()
		return var




if __name__=='__main__':
	vgg19 = Vgg19(vgg19_npy_path='./model/vgg19.npy')
	with tf.device('/cpu:0'):
		data = np.random.randn(64,448,448,3)
		inputs = tf.placeholder(tf.float32,[64,448,448,3])
		predicts = vgg19.forward(inputs)
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth=True
		init = tf.global_variables_initializer()
		sess = tf.Session(config=config)
		sess.run(init)
		output = sess.run(predicts,feed_dict={inputs:data})
		print(output.shape)
		sess.close()

