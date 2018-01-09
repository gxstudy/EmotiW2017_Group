#Useful libraries
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.learning_curve import learning_curve
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
import sys
import pickle
#from sklearn.metrics import roc_auc_score, accuracy_score
#np.set_printoptions(threshold=np.nan)

training_inputs1 = np.load('fc6_face_256_train_val_iter_3000.npy')
print training_inputs1[0].shape
print training_inputs1[1].shape
print training_inputs1[2].shape
training_inputs2 = np.load('group_vgg_train_val_iter_3200_fc6_oversample.npy')
print training_inputs2[0].shape
print training_inputs2[1].shape
print training_inputs2[2].shape
training_inputs =np.concatenate((np.concatenate((training_inputs1[0],training_inputs2[0]),axis=1),np.concatenate((training_inputs1[1],training_inputs2[1]),axis=1),np.concatenate((training_inputs1[2],training_inputs2[2]),axis=1)))
print training_inputs.shape

training_classes = np.zeros((training_inputs1[0].shape[0]+training_inputs1[1].shape[0]+training_inputs1[2].shape[0],))
training_classes[0:training_inputs1[0].shape[0]]=0
training_classes[training_inputs1[0].shape[0]:training_inputs1[0].shape[0]+training_inputs1[1].shape[0]]=1
training_classes[training_inputs1[0].shape[0]+training_inputs1[1].shape[0]:]=2
print training_classes.shape

testing_inputs1 = np.load('../vgg_faces/weight_vgg_faces_fc6.npy')
print testing_inputs1.shape

testing_inputs2 = np.load('../scene_pred/scene_vgg_fc6.npy')
print testing_inputs2.shape
testing_inputs =np.concatenate((testing_inputs1,testing_inputs2),axis=1)
testing_inputs = np.nan_to_num(testing_inputs)
print testing_inputs.shape
'''testing_classes = np.zeros((testing_inputs1[0].shape[0]+testing_inputs1[1].shape[0]+testing_inputs1[2].shape[0],))
testing_classes[0:testing_inputs1[0].shape[0]]=0
testing_classes[testing_inputs1[0].shape[0]:testing_inputs1[0].shape[0]+testing_inputs1[1].shape[0]]=1
testing_classes[testing_inputs1[0].shape[0]+testing_inputs1[1].shape[0]:]=2
print testing_classes.shape'''


#Feature Normalization before SVM
training_inputs = np.nan_to_num(training_inputs)
sc = StandardScaler()
sc.fit(training_inputs)
#val_inputs= sc.transform(val_inputs)
training_inputs= sc.transform(training_inputs)
testing_inputs = sc.transform(testing_inputs)

#lodad svm model and predict on validation set and testing set
clf = pickle.load(open('svm_faces_scene_model_2.sav', 'rb'))
clf_predict =clf.predict(training_inputs)
print('svc ovo Accuracy on validation data: %.3f' % accuracy_score(y_true=training_classes, y_pred=clf_predict))

#file_name = "svm_predict_val_label.npy"
#np.save(file_name, clf_predict)
clf_prob = clf.predict_proba(testing_inputs)
file_name = "svm_combine_face_scene_test_prob_2.npy"
np.save(file_name, clf_prob)

 

