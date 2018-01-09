% created on 1/4/2016
% created by Cindy Guo
% Build a text file corresponding to the cropped faces and it's video

% train: note there are 773 videos, but only 756 has corpped faces
% train set
clc
clear all
close all

face_path = '../data/Test_vj_faces';
face_save_path = '../data/Test_gray_vj_faces'
if ~ exist(face_save_path,'dir')
    mkdir(face_save_path);
end 
files = dir(face_path);
for i = 1 : length(files)     
    if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') 
        continue;
    end
    image_name = files(i).name;


    sub_files_path  = [face_path '/' image_name];
    sub_files = dir(sub_files_path);
    sub_face_save_path = [face_save_path '/' image_name];
    if ~ exist(sub_face_save_path,'dir')
        mkdir(sub_face_save_path);
    end 
    file_index = 0;
    for j = 1 : length(sub_files)     
        if strcmp(sub_files(j).name, '.') || strcmp(sub_files(j).name, '..') 
            continue;
        end
        face_name = [sub_files_path '/' sub_files(j).name];
        img = imread(face_name);
        img = rgb2gray(img);
        face_save_name = [sub_face_save_path '/' sub_files(j).name];

        imwrite(img,face_save_name);

    end
end



