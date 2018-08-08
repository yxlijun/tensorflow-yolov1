from __future__ import absolute_import 
from __future__ import print_function
from __future__ import division 


from yolo_solver import YoloSolver
from dataset import yoloDataset
import config as cfg 
from yoloLoss import yoloLoss

import argparse 
import tensorflow as tf 
import os 

parser = argparse.ArgumentParser()
parser.add_argument('-n','--net',type=str,default='Vgg16',choices=cfg.net_style,help='net style')
parser.add_argument('--gpu',type=int,default=1,help='train gpu')
FLAGS,unknown = parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES']= str(FLAGS.gpu)

vgg16_config = {
		'name':'Vgg16',
		'pretained_model':'./model/vgg16.npy',
		'mode':0
	}

vgg19_config = {
	'name':'Vgg19',
	'pretained_model':'./model/vgg16.npy',
	'mode':0
}

resnet50_config = {
	'name':'resnet50',
	'pretained_model':'./model/resnet',
	'mode':1
}


def choose_net(FLAGS,nets):
	for net in nets:
		if net['name'] == FLAGS.net:
			return net 
	return resnet50_config


def main():
	print('please choose net from:',cfg.net_style)
	netconfig = choose_net(FLAGS,[vgg16_config,vgg19_config,resnet50_config])
	train_dataset = yoloDataset(cfg.common_params,cfg.dataset_params,cfg.dataset_params['train_file'])
	test_dataset = yoloDataset(cfg.common_params,cfg.dataset_params,cfg.dataset_params['test_file'],train=False)
	dataset = {
		'train':train_dataset,
		'test':test_dataset
	}
	yololoss = yoloLoss(cfg.common_params,netconfig)
	solver = YoloSolver(dataset,netconfig,yololoss,cfg.common_params,cfg.solver_params)
	solver.solve()


if __name__=='__main__':
	main()