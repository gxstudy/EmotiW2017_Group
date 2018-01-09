import sys
sys.path.append('/home/xin/caffe/distribute/python')
import caffe
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

model_def = 'ResNet-152-deploy.prototxt'

model_weights ="group_train_val_face_pose_hand_resnet152_iter_8000.caffemodel"
# load the mean ImageNet image (as distributed with Caffe) for subtraction
#mu = np.load("Train_group_mean.npy")
#mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
mu = np.array([6, 9, 6])
net = caffe.Classifier(model_def, model_weights,
                       mean=mu,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))


image_dir = '../data/Test_converted'
pose_dir = '../data/group_Skeleton_face_pose_hand_valid' 

lcount = 0
for ename in sorted(os.listdir(image_dir)):
    lcount += 1
print lcount  

face_pose_hand_resnet152_preds = np.zeros((lcount,3))

num_count = 0
for iname in sorted(os.listdir(image_dir)):
    print iname
    fname = iname.split('.')
    postfix_name = fname[-1]
    truc_name = iname[0:-(len(postfix_name)+1)]
    pose_name = truc_name + '_rendered.png'
    print pose_name
    image_name = os.path.join(pose_dir,pose_name)
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
    face_pose_hand_resnet152_preds[num_count]=prediction[0]
    print face_pose_hand_resnet152_preds[num_count] 
    num_count += 1  

np.save('face_pose_hand_resnet152_preds.npy',face_pose_hand_resnet152_preds)
