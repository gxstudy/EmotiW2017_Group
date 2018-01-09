% created on 1/4/2016
% created by Xin Guo

clc
clear all
close all

face_path = '../data/Test';

save_path = '../data/Test_converted1';
if ~exist(save_path,'dir')
    mkdir(save_path)
end

files = dir(face_path);
for i = 1 : length(files)     
    if strcmp(files(i).name, '.') || strcmp(files(i).name, '..')
        continue;
    end
    image_name = files(i).name;
    C = strsplit(image_name,'.');
    suffix = C{end}
    source_frame  = [face_path '/' image_name];
    if strcmp(suffix,'jpeg')==1              
        new_image_name = [image_name(1:end-4) 'jpg' ]
        org_image = imread(source_frame);  
        save_name = [save_path '/' new_image_name];           
        imwrite(org_image, save_name);
    else
        save_name = [save_path '/' image_name];
        copyfile(source_frame,save_name);
    end


end



