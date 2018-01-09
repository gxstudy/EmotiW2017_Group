% created on 1/4/2016
% created by Cindy Guo
% Build a text file corresponding to the cropped faces and it's video

% Val: note there are 773 videos, but only 756 has corpped faces
% Val set
clc
clear all
close all

original_path = '../data/Test_converted';
skeleton_path = '../data/group_Skeleton_face_pose';
save_folder = '../data/group_Skeleton_face_pose_valid';
if ~exist(save_folder,'dir')
    mkdir(save_folder)
end

index=0;

files = dir(original_path);

for i = 1 : length(files)     
    if strcmp(files(i).name, '.') || strcmp(files(i).name, '..')
        continue;
    end
    image_name = files(i).name
    source_frame  = [original_path '/' image_name];
    org_image = imread(source_frame);
    fname = strsplit(image_name,'.')
    postfix_name = fname{end}
    fname2 = strsplit(image_name, ['.' postfix_name])
    if length(fname2)==2
        fname3 = fname2{1};
    else
        fname3='';
        for j =1:length(fname2)-1
            fname3 = [fname3 fname2{j}];
            if j~= length(fname2)-1
               fname3 = [fname3 ['.' postfix_name]]; 
            end
        end
    end
    %720x1280 fixed dimenasionality
    ratio = 720/size(org_image,1);
    width = ratio*size(org_image,2)
    skeleton = [skeleton_path '/' fname3 '_rendered.png'];
    skeleton_save_name = [save_folder '/' fname3 '_rendered.png'];

    skeleton_image = imread(skeleton);
    cut_width=floor(width);
    if cut_width >size(skeleton_image,2)
        cut_width = size(skeleton_image,2);
    end
    skeleton_image_valid = skeleton_image(:,1:cut_width,:);
    imwrite(skeleton_image_valid, skeleton_save_name);      
end



