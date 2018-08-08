## tensorflow YOLO-v1

* this is a study project, which is not exactly the same as the original [paper](https://arxiv.org/pdf/1506.02640.pdf) in the Network structure,we use vgg16 achieve 65.34% map in the voc2007test.

* The purpose that I write the yolov1 to study,and it achieve a good performance.

![](https://github.com/yxlijun/tensorflow-yolov1/blob/master/demo/result/000005_result.jpg)

![](https://github.com/yxlijun/tensorflow-yolov1/blob/master/demo/result/person_result.jpg)

## Train on voc2007+2012
| model      | map@voc2007test | 	
|------------| ----------------|
| VGG16      |	65.34%         |
| VGG19      |  66.12%         |
| Resnet50   |  65.23%         |

## 1.requirement
- opencv
- tensorflow 1.8
- numpy


## 2.prepare dataset
1. download voc2007 and voc2012 dataset
2. unzip dataset as following
	* VOCdevkit
		* VOC2007
			* Annotations
			* ImageSets
			* JPEGImages
			* SegmentationClass
			* SegmentationObject
		* VOC2012	
		
3. ```python  proprecess_pasval_voc.py```

## 3.training 
```
python train.py \\  
	--net {Vgg16||Vgg19||resnet50}
	--gpu 0
```
## 4.predict and eval
```
python predict --net {Vgg16||Vgg19||resnet50}
python eval.py
```

## result
![](https://github.com/yxlijun/tensorflow-yolov1/blob/master/demo/result/000038_result.jpg)
![](https://github.com/yxlijun/tensorflow-yolov1/blob/master/demo/result/000023_result.jpg)
![](https://github.com/yxlijun/tensorflow-yolov1/blob/master/demo/result/000022_result.jpg)
![](https://github.com/yxlijun/tensorflow-yolov1/blob/master/demo/result/000017_result.jpg)
![](https://github.com/yxlijun/tensorflow-yolov1/blob/master/demo/result/000004_result.jpg)