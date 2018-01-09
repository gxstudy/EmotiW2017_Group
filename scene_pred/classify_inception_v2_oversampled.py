import sys
sys.path.append('/home/xin/caffe/distribute/python')
import caffe
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


model_def = 'inception_v2_deploy.prototxt'

model_weights ="group_inception_v2_iter_3200.caffemodel"
# load the mean ImageNet image (as distributed with Caffe) for subtraction
#mu = np.load("Train_group_mean.npy")
#mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
mu = np.array([95, 100, 113])
net = caffe.Classifier(model_def, model_weights,
                       mean=mu,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

image_dir = '../data/Test_converted'
lcount=0  
for ename in sorted(os.listdir(image_dir)):
    lcount += 1
print lcount  

scene_inception_preds = np.zeros((lcount,3))
    
num_count = 0
for iname in sorted(os.listdir(image_dir)):
    print iname
    image_name = os.path.join(image_dir,iname)
    input_image = caffe.io.load_image(image_name)
    if len(input_image.shape)>3:
        print input_image.shape
        converted_image = input_image[0]
        print converted_image.shape
        
    else:
        converted_image=input_image
    img = caffe.io.resize_image(converted_image, (256,256), interp_order=3 )
    prediction = net.predict([img])  # predict takes any number of images, and formats them for the Caffe net automatically
    print 'predicted class:', prediction[0].argmax()
    scene_inception_preds[num_count]=prediction[0]
    print scene_inception_preds[num_count]
    num_count += 1  
np.save('scene_inception_preds.npy',scene_inception_preds)
