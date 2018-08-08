# -*- coding:utf-8 -*-
from __future__ import division 
from __future__ import print_function 
from __future__ import absolute_import 

import tensorflow as tf 
import numpy as np 
import config as cfg 


import cv2 
import os 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-n','--net',type=str,default='Vgg16',choices=cfg.net_style,help='net style')
parser.add_argument('-d','--demo',action="store_true", default=True)
parser.add_argument('-t','--test',action="store_true", default=False)
parser.add_argument('--gpu',type=int,default=1,help='choose gpu')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

print('please choose net from:',cfg.net_style)


VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
'sheep', 'sofa', 'train', 'tvmonitor')


cachedir = './cachedir'
if not os.path.exists(cachedir):
     os.mkdir(cachedir)

cachedir_predict = os.path.join(cachedir,'predict_dir')
if not os.path.exists(cachedir_predict):
    os.mkdir(cachedir_predict)

det_list = [os.path.join(cachedir_predict,file) for file in os.listdir(cachedir_predict)]
for det_class_file in det_list:
    with open(det_class_file,mode='w') as f:
        pass

Color = [[0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],[128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]]


def decoder(pred):
    grid_num = 14
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1./grid_num 
    pred = np.squeeze(pred)
    contain1 = pred[:,:,4][:,:,np.newaxis]
    contain2 = pred[:,:,9][:,:,np.newaxis]
    contain = np.concatenate((contain1,contain2),axis=-1)
    mask1 = contain>0.1
    mask2 = (contain==contain.max())
    mask = ((mask1+mask2)>0).astype(np.int32)
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i,j,b]==1:
                    box = pred[i,j,b*5:b*5+4]
                    contain_prob = pred[i,j,b*5+4]
                    xy = np.array([j,i])*cell_size
                    box[:2] = box[:2]*cell_size+xy
                    box_xy = np.zeros_like(box)
                    box_xy[:2] = box[:2]-0.5*box[2:]
                    box_xy[2:] = box[:2]+0.5*box[2:]
                    max_prob = np.max(pred[i,j,10:])
                    cls_index = np.argmax(pred[i,j,10:])
                    if float(contain_prob*max_prob)>0.1:
                        boxes.append(box_xy)
                        cls_indexs.append(cls_index)
                        probs.append(contain_prob*max_prob)
    if len(boxes)==0:
        boxes = np.zeros((1,4))
        probs = np.zeros(1)
        cls_indexs = np.zeros(1)
    else:
        boxes =np.array(boxes)
        probs = np.array(probs)
        cls_indexs = np.array(cls_indexs)
    keep = nms(boxes,probs)
    return boxes[keep],cls_indexs[keep],probs[keep]


def nms(bboxes,scores,threshold=0.5):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order)>0:
        i = order[0]
        keep.append(i)
        if len(order)==1:
            break

        xx1 = x1[order[1:]].clip(min=x1[i])
        yy1 = y1[order[1:]].clip(min=y1[i])
        xx2 = x2[order[1:]].clip(max=x2[i])
        yy2 = y2[order[1:]].clip(max=y2[i])

        w = (xx2-xx1).clip(min=0)
        h = (yy2-yy1).clip(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero()[0]
        if len(ids) == 0:
            break
        order = order[ids+1]
    return np.array(keep)


def predict_image(sess,image_name):
    image = cv2.imread(image_name)
    mean = (103.939, 116.779, 123.68)
    img = image - np.array(mean,dtype=np.float32)
    h,w, _ = img.shape
    img = cv2.resize(img,(448,448))
    img = np.reshape(img,(1,448,448,3))
    graph = tf.get_default_graph()
    prob_tensor = graph.get_tensor_by_name("output:0")
    inputs = graph.get_tensor_by_name("input:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    pred = sess.run(prob_tensor,feed_dict={inputs:img,is_training:False})
    boxes,cls_indexs,probs =  decoder(pred)

    result = []
    for i,box in enumerate(boxes):
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index) # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1,y1),(x2,y2),VOC_CLASSES[cls_index],image_name,prob])
    return result



def parse_test_file():
    image_set = []
    fs_input = tf.gfile.GFile(cfg.dataset_params['test_file'],'r')
    for line in fs_input.readlines():
        line = line.strip().split(' ')
        image_set.append(line[0])
    fs_input.close()
    return image_set

def test(sess):
    image_set = parse_test_file()
    for image_name in image_set:
    	image_id = image_name.split('/')[-1].split('.')[0]
    	result = predict_image(sess,image_name)
    	for left_up,right_bottom,class_name,_,prob in result:
		filename = os.path.join(cachedir_predict,'det_test_'+class_name+'.txt')
		with open(filename,mode='a') as f:
		    left,top,right,bottom = left_up[0],left_up[1],right_bottom[0],right_bottom[1]
		    content = image_id+' '+str(prob)+' '+str(int(left))+' '+str(int(top))+' '+str(int(right))+' '+str(int(bottom))+'\n'
		    f.write(content)

def demo(sess,path):
    image_list = [os.path.join(path,x) for x in os.listdir(path) if x.endswith('jpg') or x.endswith('png')]
    out_path = os.path.join(path,'result')
    for image_name in image_list:
        image = cv2.imread(image_name)
        result = predict_image(sess,image_name)
        for left_up,right_bottom,class_name,_,prob in result:
            color = Color[VOC_CLASSES.index(class_name)]
            cv2.rectangle(image,left_up,right_bottom,color,2)
            label = class_name+str(round(prob,2))
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1 = (left_up[0], left_up[1]- text_size[1])
            cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
            cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_name = os.path.join(out_path,os.path.basename(image_name).split('.')[0]+'_result.jpg')
        cv2.imwrite(out_name,image)


if __name__=='__main__':
    model_folder = os.path.join(cfg.solver_params['model_dir'],args.net,'saved_65_34/ckpt')
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    print(input_checkpoint)

    path = './demo'
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    saver.restore(sess, input_checkpoint)

    if args.demo:
        demo(sess,path)
    if args.test:
        test(sess)
    sess.close()