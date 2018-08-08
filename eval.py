# -*- coding:utf-8 -*-
from __future__ import division 
from __future__ import print_function 
from __future__ import absolute_import 

import tensorflow as tf 
import numpy as np 
import config as cfg 
from  voc_eval import voc_eval

import cv2 
import os 

cachedir_predict = './cachedir/predict_dir'
cachedir = './cachedir'
imageset = ('VOC2007','test')

if not os.path.exists(cachedir):
    os.mkdir(cachedir)

det_list = [os.path.join(cachedir_predict,file) for file in os.listdir(cachedir_predict)]
det_classes = list()
for file in det_list:
    classes = os.path.splitext(os.path.basename(file))[0].split('_')[-1]
    det_classes.append(classes)
    detpath = file.replace(classes,'%s')

ROOT_PATH = os.path.expanduser("~")
VOC_PATH = os.path.join(ROOT_PATH,'data','VOCdevkit',imageset[0])


annopath = os.path.join(VOC_PATH,'Annotations','%s.xml')
imagesetfile = os.path.join(VOC_PATH,'ImageSets','Main',imageset[1]+'.txt')

MAPList = list()
for classname in det_classes:
	rec,prec,ap = voc_eval(detpath,annopath,imagesetfile,classname,cachedir)
	print('%s\t AP:%.4f' %(classname,ap))
	MAPList.append(ap)

Map = np.array(MAPList)
mean_Map = np.mean(Map)
print('------ Map: %.4f' %(mean_Map))
