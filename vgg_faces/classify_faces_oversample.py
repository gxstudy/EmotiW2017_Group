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
import Image
import ImageOps

feature_len = 4096
model_def = 'vgg_group_deploy.prototxt'
model_weights ="snapshot_group_face_256_train_val_combine1_iter_3000.caffemodel"
mu = np.array([84, 95, 125])
net = caffe.Classifier(model_def, model_weights,
                       mean=mu,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))


val_dir = '../data/Test_converted'
val_face_dir = '../data/Test_combined_faces'

lcount=0
for ename in sorted(os.listdir(val_dir)):
  lcount += 1
print lcount

avg_pred_faces = np.zeros((lcount,3))
weighted_pred_faces = np.zeros((lcount,3))
fc6_whole= np.zeros((lcount,feature_len))

num_count = 0
num_undetected = 0
for iname in sorted(os.listdir(val_dir)):
  print iname
  fname = iname.split('.')
  postfix_name = fname[-1]
  fname3 = iname[0:-(len(postfix_name)+1)]
  if os.path.exists(os.path.join(val_face_dir,fname3)): 
    RGB_frames = glob.glob('%s/%s/*.jpg' %(val_face_dir,fname3))      
    predict_single_frame = np.zeros((len(RGB_frames),3))
    fc6_single_frame = np.zeros((len(RGB_frames),feature_len))
    frame_weight = np.zeros((len(RGB_frames),))
    
    index_frame=0
    for im in RGB_frames:
      input_image = caffe.io.load_image(im)
      img = caffe.io.resize_image( input_image, (256,256), interp_order=3 )
      prediction = net.predict([img], oversample=True)      
      print prediction
      predict_single_frame[index_frame, :] = prediction[0]
      fc6_single_frame[index_frame,:] = net.blobs['fc6'].data[0]
      sp = re.split('/',im)
      image_name = re.split('.jpg',sp[-1])
      image_dim = re.split('_',image_name[0])
      frame_weight[index_frame] = int(image_dim[-1])*int(image_dim[-2])
      print frame_weight[index_frame]
      index_frame = index_frame+1
    #print predict_single_frame
    avg_pred_faces[num_count,:] = np.mean(predict_single_frame,0)
    fc6_whole[num_count,:] = np.mean(fc6_single_frame,0)
    #print avg_pred_per_class[num_count,:]
    weighted_predict_prob = np.zeros((3,))
    norm_facter = np.sum(frame_weight)
    print norm_facter
    for i in range(len(RGB_frames)):
      weighted_predict_prob = weighted_predict_prob + frame_weight[i]*predict_single_frame[i]/norm_facter 
    weighted_pred_faces[num_count,:] = weighted_predict_prob      
    print avg_pred_faces[num_count,:] 
    print weighted_pred_faces[num_count,:]
    print fc6_whole[num_count,:]
    print 'Avg predicted class:', avg_pred_faces[num_count,:] .argmax()
    print 'Weight predicted class:', weighted_pred_faces[num_count,:] .argmax()
    num_count += 1 
  else:
    print('No faces detected in this image')
    #sys.exit()
    avg_pred_faces[num_count,:] = [0,0,0]
    weighted_pred_faces[num_count,:] = [0,0,0] 
    fc6_whole[num_count,:]=np.zeros((1,feature_len))
    print avg_pred_faces[num_count,:] 
    print weighted_pred_faces[num_count,:]
    print fc6_whole[num_count,:]
    print 'Avg predicted class:', avg_pred_faces[num_count,:] .argmax()
    print 'Weight predicted class:', weighted_pred_faces[num_count,:] .argmax()
    num_count += 1  
    num_undetected+=1
#np.save('avg_face_256_combine1_iter_2000_oversample_V2.npy',avg_pred_faces)
np.save('weight_vgg_faces_probs.npy',weighted_pred_faces)
np.save('weight_vgg_faces_fc6.npy',fc6_whole)
print num_undetected

    
