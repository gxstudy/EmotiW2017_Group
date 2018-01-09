# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 20:00:17 2016

@author: fanyin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:54:34 2016

@author: fanyin
"""
import os
import numpy 
from sklearn import svm
#import sys
#sys.path.insert(0, '../caffe/python/')
#import caffe
import sys
sys.path.append('/home/xin/caffe/distribute/python')
import caffe
import time
import shutil
import string 
import re
import PIL
from PIL import Image
#sequence_num=128


def face_fliter_qiyi(gray_image_dir,color_image_dir, color_save_dir):  
    prototxt = './models/face_fliter_deploy.prototxt'
    caffemodel = './models/face_fliter_7.caffemodel'
    caffe.set_mode_gpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', numpy.load('./models/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    filtered_out_number=0
    filtered_out_images = [] 

    if os.path.exists(color_save_dir) == False:
        os.mkdir(color_save_dir)
    for subdir in sorted(os.listdir(gray_image_dir)):
        cur_dir = os.path.join(gray_image_dir, subdir)
        cur_save_dir = os.path.join(color_save_dir, subdir)
	if os.path.exists(cur_save_dir) == False:
	    os.mkdir(cur_save_dir)
	face_count = 0
	for image_file in sorted(os.listdir(cur_dir)):
	    if image_file[-4:] == ".jpg":
		full_dir = os.path.join(cur_dir, image_file)
		net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(full_dir))
		out = net.forward()
		if out['prob'].argmax() == 1 and out['prob'][0,out['prob'].argmax()] >= 0.90:
		    color_dir= os.path.join(color_image_dir,subdir, image_file)
		    face_count += 1
		    shutil.copy(color_dir, os.path.join(cur_save_dir, image_file))
	if face_count == 0:
	    print('No faces is detected in this folder')
	    os.rmdir(cur_save_dir)
	    filtered_out_images.append(cur_save_dir)
            filtered_out_number+=1
    print filtered_out_number
    print filtered_out_images  
                
gray_image_dir = "../data/Test_gray_vj_faces"
color_image_dir = "../data/Test_vj_faces"
color_save_dir = "../data/Test_filtered_vj_faces"
face_fliter_qiyi(gray_image_dir,color_image_dir,color_save_dir)

     
