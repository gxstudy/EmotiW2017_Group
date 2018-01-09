% created on 1/4/2016
% created by Cindy Guo
% extract path of images into a list

clc
clear all
close all

face_path = '../data/Test_converted';
fileID1 = fopen('test_converted_list.txt','w');
index=0;


files = dir(face_path);
for i = 1 : length(files)     
    if strcmp(files(i).name, '.') || strcmp(files(i).name, '..')
        continue;
    end
    image_name = files(i).name
    source_frame  = [face_path '/' image_name]   
    fprintf(fileID1,'%s\n',source_frame(3:end));
end


