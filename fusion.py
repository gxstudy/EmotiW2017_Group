import os
import numpy as np
import re 

class_dict = {}
class_dict[0] = 'Negative'
class_dict[1] = 'Neutral'
class_dict[2] = 'Positive'

def compute_fusion(pred_1, pred_2, pred_3,pred_4, pred_5, pred_6, p1, p2,p3, p4, p5):
    print 1-p1-p2-p3-p4-p5
    fusion_pred = p1*pred_1 + p2*pred_2 + p3*pred_3 + p4*pred_4 + p5*pred_5 +(1-p1-p2-p3-p4-p5)*pred_6
    return fusion_pred 

svm_combine_face_scene_test_prob = np.load('./scene_face_combine_pred/svm_combine_face_scene_test_prob_2.npy')
print svm_combine_face_scene_test_prob.shape
print svm_combine_face_scene_test_prob

#weight_vgg_faces_unpositive = np.load('./vgg_faces/weight_vgg_faces_unpositive.npy')
#print weight_vgg_faces_unpositive.shape
#print weight_vgg_faces_unpositive

weight_vgg_faces_positive = np.load('./vgg_faces/weight_vgg_faces_positive.npy')
print weight_vgg_faces_positive.shape
print weight_vgg_faces_positive

scene_inception_preds=np.load('./scene_pred/scene_inception_preds.npy')
print scene_inception_preds.shape
print scene_inception_preds

weight_vgg_faces_probs = np.load('./vgg_faces/weight_vgg_faces_probs.npy')
print weight_vgg_faces_probs.shape
print weight_vgg_faces_probs

#scene_vgg_preds = np.load('./scene_pred/scene_vgg_preds.npy')
#print scene_vgg_preds.shape
#print scene_vgg_preds

face_pose_inception_preds = np.load('./skeleton_pred/face_pose_inception_preds.npy')
print face_pose_inception_preds.shape
print face_pose_inception_preds

face_pose_hand_resnet152_preds = np.load('./skeleton_pred/face_pose_hand_resnet152_preds.npy')
print face_pose_hand_resnet152_preds.shape
print face_pose_hand_resnet152_preds

pred_combine = compute_fusion(svm_combine_face_scene_test_prob,weight_vgg_faces_positive,scene_inception_preds, weight_vgg_faces_probs ,face_pose_inception_preds,face_pose_hand_resnet152_preds,0.05,0.20,0.25,0.255,0.15)
print pred_combine.shape
print pred_combine
np.save('compute_fusion_submitted_6.npy', pred_combine)
pred_label = np.argmax(pred_combine ,1)
print pred_label.shape
    

save_dir = './6 - UD-GPB - Group'
os.mkdir(save_dir)

image_dir = './data/Test_converted'
icount=0
for iname in sorted(os.listdir(image_dir)):
    print iname
    print icount
    fname = iname.split('.')
    postfix_name = fname[-1]
    truc_name = iname[0:-(len(postfix_name)+1)]
    save_name = truc_name + '.txt'
    print pred_combine[icount]
    label = pred_label[icount]
    print label
    fp = open(os.path.join(save_dir, save_name), 'w')
    fp.write(class_dict[label])
    fp.close()   
    icount +=1
