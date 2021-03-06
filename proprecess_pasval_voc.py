import os
import numpy as np 
import tensorflow as tf
import config as cfg 

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
classes_to_id = dict(zip(classes_name,range(len(classes_name))))

ROOT_PATH = os.path.expanduser("~")
YOLO_PATH = os.path.abspath('./')
VOC_PATH = os.path.join(ROOT_PATH,'data','VOCdevkit')
VOC_PATH_2007 = os.path.join(VOC_PATH,"VOC2007")
VOC_PATH_2012 = os.path.join(VOC_PATH,"VOC2012")
DATA_PATH = os.path.join(YOLO_PATH,'data')
output_path = os.path.join(DATA_PATH,'pasvoc_0712.txt')
output_test_path = os.path.join(DATA_PATH,'pasvoc_0712_test.txt')
if not tf.gfile.Exists(DATA_PATH):
	tf.gfile.MakeDirs(DATA_PATH)

difficult_cut = True
def parse_xml(xml_path):
	tree = ET.parse(xml_path)
	folder =  tree.find('folder').text
	filename = tree.find('filename').text
	imagepath = os.path.join(VOC_PATH,folder,'JPEGImages',filename)
	labels = []
	bbox = ['xmin','ymin','xmax','ymax']
	for obj in tree.findall('object'):
		difficult = int(obj.find('difficult').text)
		if difficult and difficult_cut:
			continue
		classes = obj.find('name').text
		bndbox = obj.find('bndbox')
		for locmark in bbox:
			labels.append(bndbox.find(locmark).text)
		labels.append(classes_to_id[classes])
	return imagepath,labels

		
	
def convert_to_string(imagepath,labels):
	output_string = imagepath
	for info in labels:
		output_string+=' '
		output_string+=str(info)
	output_string+='\n'
	return output_string

def main():
	out_file = tf.gfile.GFile(output_path,'w')
	out_test_file = tf.gfile.GFile(output_test_path,'w')
	xml_dir_2007 = os.path.join(VOC_PATH_2007,'Annotations')
	xml_dir_2012 = os.path.join(VOC_PATH_2012,'Annotations')

	xml_list_2007 = [os.path.join(xml_dir_2007,file) for file in os.listdir(xml_dir_2007)]
	xml_list_2012 = [os.path.join(xml_dir_2012,file) for file in os.listdir(xml_dir_2012)]
	xml_list = xml_list_2007
	xml_list.extend(xml_list_2012)
	train_set = cfg.dataset_params['train_set']
	train_file_list = []
	for voc in train_set:
		voc = voc.split('_')
		file = os.path.join(VOC_PATH,voc[-1],'ImageSets/Main',voc[0]+'.txt')
		train_file_list.append(file)
	
	train_id = []
	for file in train_file_list:
		fs = tf.gfile.GFile(file,'r')
		for line in fs.readlines():
			line = line.strip()
			train_id.append(str(line))


	test_set = cfg.dataset_params['test_set']
	test_file_list = []
	for voc in test_set:
		voc = voc.split('_')
		file = os.path.join(VOC_PATH,voc[-1],'ImageSets/Main',voc[0]+'.txt')
		test_file_list.append(file)
	test_id = []
	for file in test_file_list:
		fs = tf.gfile.GFile(file,'r')
		for line in fs.readlines():
			line = line.strip()
			test_id.append(str(line))

	for xml_path in xml_list:
		xml_id = os.path.splitext(os.path.basename(xml_path))[0]
		if xml_id in train_id:
			try:
				imagepath,labels = parse_xml(xml_path)
				output_string = convert_to_string(imagepath,labels)
				out_file.write(output_string)
			except Exception:
				pass
		if xml_id in test_id:
			try:
				imagepath,labels = parse_xml(xml_path)
				output_string = convert_to_string(imagepath,labels)
				out_test_file.write(output_string)
			except Exception:
				pass



	out_test_file.close()
	out_file.close()
if __name__=='__main__':
	main()
