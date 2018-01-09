% created on 1/4/2016
% created by Cindy Guo
% Build a text file corresponding to the cropped faces and it's video

% train: note there are 773 videos, but only 756 has corpped faces
% train set
clc
clear all
close all
addpath('./violajones')


face_path = '../data/Test_converted';
target_path = '../data/Test_vj_faces';
if ~exist(target_path, 'dir')
    mkdir(target_path);
end
fileID2 = fopen('group_test_vj_undetected_name.txt','w');
files = dir(face_path);
for i = 1 : length(files)     
    if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') 
        continue;
    end
    image_name = files(i).name;
    source_frame  = [face_path '/' image_name];
    image = imread(source_frame);
    faceDetector = vision.CascadeObjectDetector;
    bboxes = step(faceDetector, image);
    target_image_folder = [target_path '/' image_name(1:end-4)];
    
    if size(bboxes,1)>0
        if ~exist(target_image_folder, 'dir')
            mkdir(target_image_folder);
        end
        for index=1:1:size(bboxes,1)
            image_new_name = [target_image_folder '/V_' num2str(bboxes(index,1)) '_' num2str(bboxes(index,2)) '_' num2str(bboxes(index,3)) '_' num2str(bboxes(index,4)) '.jpg'];
            face = image(bboxes(index,2):(bboxes(index,2)+round(bboxes(index,4))),bboxes(index,1):(bboxes(index,1)+round(bboxes(index,3))),:);
            imwrite(face, image_new_name);
        end
    else
        fprintf(fileID2,'%s\n',source_frame(3:end));
    end
end

