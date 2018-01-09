# -*- coding: utf-8 -*-
"""
Created on Fri May 20 18:34:56 2016

@author: fanyin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:49:18 2016

@author: fanyin
"""

import os
import numpy as np
from sklearn import svm
import sys
import string
import re
from pylab import * 
import glob
import sys
sys.path.append('/home/xin/caffe/distribute/python')
import caffe
caffe.set_mode_gpu()
#caffe.set_device(1)
import shutil

org_data_dir = '../data/Test_converted'
aligned_faces_dir = '../data/Test_faces_aligned'
vj_faces_dir = '../data/Test_filtered_vj_faces'
combine_dir = '../data/Test_combined_faces'



for iname in sorted(os.listdir(org_data_dir)):
    fname = iname.split('.')
    suffix = fname[-1]
    fname1 = iname[0:-(len(suffix)+1)]
    undetected_image = []
    undetected_image_number = 0

    align_flag=0
    vj_flag = 0
    if os.path.exists(os.path.join(combine_dir,fname1)) == False:
        os.makedirs(os.path.join(combine_dir,fname1)) 

    if os.path.exists(os.path.join(aligned_faces_dir, fname1)):         
        RGB_frames_align = glob.glob('%s/%s/*.jpg' %(aligned_faces_dir,fname1))
        align_flag = 1
    if os.path.exists(os.path.join(vj_faces_dir, fname1)): 
        RGB_frames_vj = glob.glob('%s/%s/*.jpg' %(vj_faces_dir,fname1))
        vj_flag = 1
    if align_flag==1:
        for im_align in RGB_frames_align:
            image_name = im_align.split('/')[-1]
            shutil.copy(im_align, os.path.join(combine_dir,fname1,image_name))
        if vj_flag==1:       
	    for im_vj in RGB_frames_vj:
		sp_vj = re.split('/',im_vj)
		vj_name = re.split('.jpg',sp_vj[-1])
		vj_dim = re.split('_',vj_name[0]) 
		center_image_x = int(vj_dim[-4])+int(0.5*int(vj_dim[-2]))
		center_image_y = int(vj_dim[-3])+int(0.5*int(vj_dim[-1]))
		#print center_image_x,center_image_y
		vj_size = int(vj_dim[-2])*int(vj_dim[-1])
		save_or_delete_flag=0
		for im_align in RGB_frames_align:
		    sp_align = re.split('/',im_align)
		    align_name = re.split('.jpg',sp_align[-1])
		    align_dim = re.split('_',align_name[0])
		    #print align_dim 
		    align_size = int(align_dim[-2])*int(align_dim[-1])

		    if center_image_x>=int(align_dim[-4]) and center_image_x<=(int(align_dim[-4])+int(align_dim[-2])) and center_image_y>=int(align_dim[-3]) and center_image_y<=(int(align_dim[-3])+int(align_dim[-1])): 
			save_or_delete_flag=1
			#print save_or_delete_flag
		if save_or_delete_flag==0: 
		    shutil.copy(im_vj, os.path.join(combine_dir,fname1,sp_vj[-1]))
	      
    if align_flag==0:
        if vj_flag==1: 
            for im_vj in RGB_frames_vj:
              sp_vj = re.split('/',im_vj)
              shutil.copy(im_vj, os.path.join(combine_dir,fname1,sp_vj[-1]))     
        else:
            os.rmdir(os.path.join(combine_dir,fname1)) 
            undetected_image.append(fname1)
            undetected_image_number +=1 

print "Undetected image number is " 
print  undetected_image_number
print "Undetected image names are " 
print  undetected_image
    
