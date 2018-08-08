# -*- coding:utf-8 -*-
from __future__ import absolute_import 
from __future__ import print_function
from __future__ import division 


import numpy as np 
import tensorflow as tf 
import os 
import time 
import datetime
import cPickle
from vgg16 import Vgg16
from vgg19 import Vgg19
from resnet import resnet50

class YoloSolver(object):
	"""docstring for YoloSolver"""
	def __init__(self, dataset,netconfig,loss,common_params,solver_params):
		super(YoloSolver, self).__init__()
		self.moment = solver_params['moment']
		self.learning_rate = solver_params['learning_rate']
		self.batch_size = common_params['batch_size']
		self.height,self.width = common_params['image_size'],common_params['image_size']
		self.grid_num = common_params['output_size']
		self.num_steps = common_params['num_steps']
		self.display_step = solver_params['display']

		self.netname = netconfig['name']
		self.pretained_model = netconfig['pretained_model']
		self.mode = netconfig['mode']
		self.yololoss = loss

		self.train_dataset = dataset['train']
		self.test_dataset = dataset['test']

		self.model_dir = os.path.join(solver_params['model_dir'],self.netname,'ckpt')
		if not tf.gfile.Exists(self.model_dir):
			tf.gfile.MakeDirs(self.model_dir)
		self.model_name = os.path.join(self.model_dir,'yolomodel.ckpt')
		self.model_exist = tf.gfile.Exists(os.path.join(self.model_dir,'checkpoint'))

		self.best_model_dir = os.path.join(solver_params['model_dir'],self.netname,'best')
		if not tf.gfile.Exists(self.best_model_dir):
			tf.gfile.MakeDirs(self.best_model_dir)
		self.best_model_name = os.path.join(self.best_model_dir,'best.ckpt')

		step_path = os.path.join(solver_params['model_dir'],self.netname,'step_pkl')
		if not tf.gfile.Exists(step_path):
			tf.gfile.MakeDirs(step_path)
		self.step_file = os.path.join(step_path,'step.pkl')
		self.step_exist = tf.gfile.Exists(self.step_file)

		self.contruct_graph()

	def contruct_graph(self):
		tf.set_random_seed(1)
		self.cur_step = 0
		if self.step_exist:
			step_info = cPickle.load(open(self.step_file,'r'))
			self.cur_step = step_info['step']
		self.global_step = tf.Variable(self.cur_step, trainable=False)
		self.images = tf.placeholder(tf.float32, (None, self.height, self.width, 3),name='input')
		self.targets = tf.placeholder(tf.float32, (None, self.grid_num,self.grid_num, 30),name='target')
		self.is_training = tf.placeholder_with_default(False,None,name='is_training')

		if self.mode==0:
			self.net = eval(self.netname)(self.pretained_model,self.is_training)
		else:
			self.net = eval(self.netname)(self.is_training)

		self.predicts = self.net.forward(self.images)
		self.total_loss,self.average_iou,self.object_num = self.yololoss.forward(self.predicts,self.targets)
		self.learning_rate =  tf.train.exponential_decay(self.learning_rate,self.global_step,120000,0.1,staircase=True)

		optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.moment)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.train_op = optimizer.minimize(self.total_loss,global_step=self.global_step)


	def solve(self):
		var_list = tf.trainable_variables()
		g_list = tf.global_variables()
		bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
		bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
		var_list += bn_moving_vars
		init = tf.global_variables_initializer()

		if self.mode==1:
			self.saver1 = tf.train.Saver(self.net.variables_to_restore)
			
		saver = tf.train.Saver(var_list=var_list)

		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth =True
		sess = tf.Session(config=config)
		sess.run(init)

		self.min_loss = np.inf
		if self.mode==1:
			checkpoint = tf.train.get_checkpoint_state(self.pretained_model)
			input_checkpoint = checkpoint.model_checkpoint_path
			self.saver1.restore(sess,input_checkpoint)
		if self.model_exist:
			saver.restore(sess,tf.train.latest_checkpoint(self.model_dir))
		losses = 0
		ious = 0
		for step in range(self.cur_step,self.num_steps):
			start_time = time.time()
			images,targets = self.train_dataset.batch()
			_,loss_value,iou_value,object_num,lr = sess.run([self.train_op,self.total_loss,self.average_iou,self.object_num,self.learning_rate],feed_dict={self.images:images,self.targets:targets,self.is_training:True})
			losses+=loss_value
			ious+=iou_value
			duration = time.time() - start_time
			assert not np.isnan(loss_value) ,'Model diverged with loss = Nan'

			if step % self.display_step:
				avg_loss = losses / (step+1)
				avg_iou = ious / (step+1)
				print('%s || step :%d ||learning_rate=%.5f ||loss=%.4f || average_iou = %.4f || object_num = %d' %(datetime.datetime.now(),
					step,lr,avg_loss,avg_iou,object_num))

			if step % 5000 == 0:
				step_info = {'step':step}
				cPickle.dump(step_info,open(self.step_file,'wb'))
				saver.save(sess, self.model_name, global_step=step)

			if (step+1) % self.train_dataset.num_batch_per_epoch==0:
				test_losses = 0.0
				for i in range(self.test_dataset.num_batch_per_epoch):
					test_images,test_targets = self.test_dataset.batch()
					loss = sess.run(self.total_loss,feed_dict={self.images:test_images,self.targets:test_targets,self.is_training:False})
					test_losses+=loss
				test_avg_loss = test_losses / self.test_dataset.num_batch_per_epoch
				print('test loss %.4f' %(test_avg_loss))
				if test_avg_loss<self.min_loss:
					saver.save(sess,self.best_model_name, global_step=step)
					self.min_loss = test_avg_loss
			step+=1

		sess.close()



