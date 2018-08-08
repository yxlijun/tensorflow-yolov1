# -*- coding:utf-8 -*-
common_params = {
	'image_size':448,
	'batch_size':6,
	'output_size':14,
	'num_steps':450000,
	'boxes_per_cell':2,
	'num_classes':20,
	'object_scale':1.0,
	'noobject_scale':0.5,
	'class_scale':1.0,
	'coord_scale':5.0,
}


dataset_params = {
	'train_file':'./data/pasvoc_0712.txt',
	'train_set':['trainval_VOC2007','trainval_VOC2012'],
	'test_set':['test_VOC2007'],
	'test_file':'./data/pasvoc_0712_test.txt'
}



solver_params = {
	'learning_rate':0.001,
	'moment':0.9,
	'model_dir':'record/train',
	'display':10
}


net_style=['Vgg16','Vgg19','resnet50']