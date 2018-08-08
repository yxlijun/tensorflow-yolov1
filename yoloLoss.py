from __future__ import division 
from __future__ import print_function 
from __future__ import absolute_import 

import tensorflow as tf 
import numpy as np 
import config as cfg 

class yoloLoss(object):
	"""docstring for yoloLoss"""
	def __init__(self, common_params,netconfig):
		super(yoloLoss, self).__init__()
		self.object_scale = common_params['object_scale']
		self.noobject_scale = common_params['noobject_scale']
		self.class_scale = common_params['class_scale']
		self.coord_scale = common_params['coord_scale']
		self.batch_size = common_params['batch_size']
		self.mode = netconfig['mode']

	def compute_iou(self,boxes1,boxes2):
		boxes1 = tf.stack([boxes1[:,0]-0.5*boxes1[:,2],boxes1[:,1]-0.5*boxes1[:,3],boxes1[:,0]+0.5*boxes1[:,2],boxes1[:,1]+0.5*boxes1[:,3]],axis=-1)
		boxes2 = tf.stack([boxes2[:,0]-0.5*boxes2[:,2],boxes2[:,1]-0.5*boxes2[:,3],boxes2[:,0]+0.5*boxes2[:,2],boxes2[:,1]+0.5*boxes2[:,3]],axis=-1)

		lu = tf.maximum(boxes1[:,0:2],boxes2[:,0:2])
		rd = tf.minimum(boxes1[:,2:],boxes2[:,2:])

		intersection = rd-lu
		inter_square = intersection[:,0] * intersection[:,1]

		mask = tf.cast(intersection[:,0] > 0, tf.float32) * tf.cast(intersection[:,1] > 0, tf.float32)

		inter_square = mask * inter_square

		square1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
		square2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])

		return inter_square/(square1 + square2 - inter_square + 1e-6)



	def forward(self,pred_tensor,target_tensor):
		N = pred_tensor.shape.as_list()[0]
		coo_mask = target_tensor[:,:,:,4]>0
		object_num = tf.reduce_sum( tf.cast(coo_mask,tf.int32))
		noo_mask = tf.cast(1-tf.cast(coo_mask,dtype=tf.int32),dtype=tf.bool)
		
		coo_mask = tf.expand_dims(coo_mask,axis=-1)
		coo_mask = tf.tile(coo_mask,[1,1,1,30])
		noo_mask = tf.expand_dims(noo_mask,axis=-1)
		noo_mask = tf.tile(noo_mask,[1,1,1,30])

		coo_pred = tf.reshape(tf.boolean_mask(pred_tensor,coo_mask),(-1,30))

		box_pred = tf.reshape(coo_pred[:,:10],(-1,5))
		class_pred = coo_pred[:,10:]

		coo_target = tf.reshape(tf.boolean_mask(target_tensor,coo_mask),(-1,30))
		box_target = tf.reshape(coo_target[:,:10],(-1,5))
		class_target = coo_target[:,10:]

		noo_pred = tf.reshape(tf.boolean_mask(pred_tensor,noo_mask),(-1,30))
		noo_target = tf.reshape(tf.boolean_mask(target_tensor,noo_mask),(-1,30))

		noo_pred_mask = tf.zeros_like(noo_pred)
		part1 = noo_pred_mask[:,0:4]
		part2 = tf.ones_like(tf.reshape(noo_pred_mask[:,4],(-1,1)))
		part3 = noo_pred_mask[:,5:9]
		part4 = tf.identity(part2)
		part5 = noo_pred_mask[:,10:]
		noo_pred_mask = tf.concat([part1,part2,part3,part4,part5],axis=-1)

		noo_pred_mask = tf.cast(noo_pred_mask,tf.bool)
		noo_pred_c = tf.boolean_mask(noo_pred,noo_pred_mask)
		noo_target_c = tf.boolean_mask(noo_target,noo_pred_mask)
		nooobj_loss = tf.nn.l2_loss(noo_target_c - noo_pred_c)

		box_target_iou = self.compute_iou(box_pred[:,0:4],box_target[:,0:4])
		box_target_iou = tf.reshape(box_target_iou,(-1,2))

		coo_response_mask = tf.contrib.framework.argsort(box_target_iou,axis=-1)
		coo_response_mask = tf.reshape(coo_response_mask,(-1,1))
		coo_response_mask = tf.tile(coo_response_mask,[1,5])
		coo_response_mask = tf.cast(coo_response_mask,tf.bool)
		box_target_iou = tf.reshape(box_target_iou,(-1,1))
		box_target_iou = tf.tile(box_target_iou,[1,5])
		box_target_iou = tf.cast(box_target_iou,tf.float32)

		box_pred_response = tf.reshape(tf.boolean_mask(box_pred,coo_response_mask),(-1,5))
		box_target_response_iou = tf.reshape(tf.boolean_mask(box_target_iou,coo_response_mask),(-1,5))
		box_target_response = tf.reshape(tf.boolean_mask(box_target,coo_response_mask),(-1,5))
		contain_loss = tf.nn.l2_loss(box_pred_response[:,4]-box_target_response_iou[:,4])
		loc_loss = tf.nn.l2_loss(box_pred_response[:,:2]-box_target_response[:,:2])+tf.nn.l2_loss(tf.sqrt(box_pred_response[:,2:4])-tf.sqrt(box_target_response[:,2:4]))

		not_contain_loss = tf.nn.l2_loss(box_pred_response[:,4]-box_target_response[:,4])

		class_loss = tf.nn.l2_loss(class_pred-class_target)


		var_list = tf.get_collection('losses') if self.mode==0 else tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		var_loss = tf.add_n(var_list)
		average_iou = tf.reduce_mean(box_target_response_iou[:,4])
		return (self.coord_scale*loc_loss+contain_loss+not_contain_loss+self.noobject_scale*nooobj_loss+self.class_scale*class_loss+var_loss)/self.batch_size,average_iou,object_num
		
if __name__ =='__main__':
	yololoss = yoloLoss(cfg.common_params)
	pred = np.random.randn(16,14,14,30)
	target = np.random.randn(16,14,14,30)

	pred_tensor = tf.placeholder(tf.float32,[16,14,14,30])
	target_tensor = tf.placeholder(tf.float32,[16,14,14,30])
	predict = yololoss.forward(pred_tensor,target_tensor)

	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth=True
	init = tf.global_variables_initializer()
	sess = tf.Session(config=config)
	sess.run(init)
	output = sess.run(predict,feed_dict={pred_tensor:pred,target_tensor:target})
	print(output)
	sess.close()
