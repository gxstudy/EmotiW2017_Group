# EmotiW2017_Group
This repository contains the code of the paper "Group-Level Emotion Recognition using Deep Models on Image Scene, Faces, and Skeletons", which scored the second place in Group-level Emotion Recognition sub-challenge of EmotiW2017. Paper is available at https://dl.acm.org/citation.cfm?id=3143017&CFID=851331041&CFTOKEN=34594234

EmotiW2017 Challenge website: https://sites.google.com/site/emotiwchallenge/

################################# General ##################################
1. Source code provided here doesn't include the testing data, users need to add the testing data from Group Affect Database 2.0 into the data folder. 
2. Source code provided here doesn't include preprocessing libriaries, users need to download and install them as instructed by those packages. 
3. Caffemodels download link: https://drive.google.com/drive/folders/1tXVIw5k4RAVuLDVsD3bsUs4MaVxWT16w?usp=sharing, please put the models into corresponding folder after downloading. 
4. Source code provided here considers more on functionality and reuse of authors' resources, so it includes code on both python and matlab, it can surely be realized by using either python or matlab and be more efficient by users.

################################# Prerequisites ##################################
1. ubuntu 16.04
2. Caffe: https://github.com/BVLC/caffe  (with cuda installed)
3. Matlab R2015a

################################# Preprocesssing ##############################
1. Make sure all the images are ended with .jpg or .png by the following steps:
    1) Copy test data to ./data folder. Double check wheather there is ambiguity in 2 files' names(family-get-together-002.jpg and family-get-together-002.png) in the test data. If so, rename family-get-together-002.png to family-get-together-002_1.png. 
    2) If the folder name of testing images is Distribution, rename the folder to be Test. 
    3) cd preprocesing
    4) Convert .jpeg files to .jpg files using convert_jpeg2jpg.m (run in Matlab), the resulting test data folder is Test_converted1.
    5) Convert .JPG files to .jpg files using convert_JPG2jpg.m, the resulting test data folder is Test_converted2.
    6) Manually convert "groupphoto3617.jpg" in folder Test_converted2 to "groupphoto3617.png" file because it's essentially a png file, if it is with .jpg suffix, it cannot be readed by some of our third-party preprocessing programs. Similarly, convert "screen shot 2017-02-23 at 164431.png" to "screen shot 2017-02-23 at 164431.jpg"; "screen shot 2017-02-27 at 143429.png" to screen shot "2017-02-27 at 143429.jpg"; "manny-pacquiao.png" to "manny-pacquiao.jpg"; Change the name of "Prime_Minister_Narendra_Modi_with_French_President_FranЗois_Hollande_at_the_G20_Summit.jpg" to "Prime_Minister_Narendra_Modi_with_French_President_Francis_Hollande_at_the_G20_Summit.jpg"since Dr. Dhall pointed out that this name contains unvalid Chinese Character. Convert files directly in the Test_converted2 folder, then rename the resulting folder to be Test_converted. 

2. Extract faces using methods from https://github.com/dougsouza/face-frontalization
    1) Install it to ./preprocessing/face_frontalization, you may also need to install its dependencies such as Dlib, OpenCV and ScoPy required by face_frontalization package. 
    2) Under preprocessing folder, run read_image2list.m to read frame list using Matlab.
    3) Open a terminal under folder preprocessing, detect and align face by: python detect_aligned_faces.py, the aligned faces will be stored in folder ./data/Test_faces_aligned. Note that you may need to copy folder dlib_models and folder frontalization_models from face forntalization package into ./preprocessing folder. 


3. Extract faces using viola-jones
    1) Install dependencies for Viola Jones Object Detection as instructed by https://www.mathworks.com/matlabcentral/fileexchange/29437-viola-jones-object-detection?focused=5171437&tab=function. We installed it under ./preprocessing/violajones.
    2) Run detect_vj_faces.m using Matlab. 
    3) The face filter model is from https://github.com/lidian007/EmotiW2016, the one called face_fliter_7.caffemodel downloaded from https://drive.google.com/open?id=0B6UurPOQfmP0R004bEdVbER1OFk. Since it's a model better working on graylevel images, so we convert the faces into gray level ones that share the same image name, then use the gary level ones to filter out the RGB-ones which are not faces. 
    4) Extract corresponding gray level faces by runing rgb2gray_vj_faces.m. 
    5) Filter out the non-faces by running: python face_filter.py 
  
4. Combine faces: python faces_combine.py
   The rule is as follows, put together faces detected by both methods, and keep the aligned faces if the same face is detected by both face detection methods.


5. Human skeleton features extraction using openpose: https://github.com/CMU-Perceptual-Computing-Lab/openpose
    1) Install openpose into folder ./preprocessing/openpose-master.
    2) Copy all the files from openpose_master/models into ./preprocessing/models folder.
    3) bash extract_skeletons_faces_poses.sh
    4) bash extract_skeletons_faces_poses_hands.sh
    5) Since the extracted images are of different dimensions from original images, so we convert each image back to its original dimensions using: restore_image_dimensions_withHand.m and restore_image_dimensions_withOutHand.m


################################# Extract predictions ##############################
1. Extract predictions for vgg faces
    1) cd vgg_faces
    2) Extract prediction of vgg face classifiers and fc6 features using python classify_faces_oversample.py,  
    3) Two additional face classifiers, positive only predictor and non-positive only predictor, are extracted using python convert_face_preditions.py
    4) cd ..

2. Extract scene preditions   
    1) cd scene_pred
    2) python classify_inception_v2_oversampled.py
    3) python classify_vgg_group_oversample.py
    4) cd ..


3. Extract skeleton predictions
    1) cd skeleton_pred
    2) python classify_inception_v2_group_face_pose_oversample.py
    3) python classify_resnet152_group_face_pose_hand_oversample.py

4. Extract scene + face combined svm predictions
    1) cd scene_face_combine_pred
    2) python svm_predict_2.py
    3) cd ..

5. python fusion.py, the resulting label will be saved in 6 - UD-GPB - Group.

################################# Citations ##############################
1. Please cite the following paper if it helps your research:
Group-Level Emotion Recognition using Deep Models on Image Scene, Faces, and Skeletons - Xin Guo, Luisa Polania and Kenneth Barner.

Bibtex:
@inproceedings{Guo:2017:GER:3136755.3143017,
 author = {Guo, Xin and Polan\'{\i}a, Luisa F. and Barner, Kenneth E.},
 title = {Group-level Emotion Recognition Using Deep Models on Image Scene, Faces, and Skeletons},
 booktitle = {Proceedings of the 19th ACM International Conference on Multimodal Interaction},
 series = {ICMI 2017},
 year = {2017},
 isbn = {978-1-4503-5543-8},
 location = {Glasgow, UK},
 pages = {603--608},
 numpages = {6},
 url = {http://doi.acm.org/10.1145/3136755.3143017},
 doi = {10.1145/3136755.3143017},
 acmid = {3143017},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {Decision fusion, Deep learning, EmotiW 2017, Emotion Recognition, Group level happiness prediction, Multi-model},
}

2. Please also cite corresponding reference papers if you use any of them related to this work. 
