close all
clear
clc

rootdir='./gt'; 
saveddir='./gt_crop';
subdir=dir(rootdir);

qmkdir(saveddir);
for i=1:length(subdir)
    subdirpath=fullfile(rootdir,subdir(i).name,'*.png');
    images=dir(subdirpath);
    for j=1:length(images)
        ImageName=fullfile(rootdir,subdir(i).name,images(j).name);
        [ImageData, map] = imread(ImageName);
        
        sz = size(ImageData);
        x = floor(sz(2)/2);
        y = floor(sz(1)/2);
        
        %%%  crop to multiple of
        high = floor(sz(1)/16) * 16;
        width = floor(sz(2)/16) * 16;
        %%%  crop to uniform size (512)
        %         high = 512;
        %         width = 512;
        
        patch = ImageData(y-(floor(high/2))+1:y+(floor(high/2)), x-(floor(width/2))+1:x+(floor(width/2)), :);
        
        savedname=fullfile(saveddir, strcat(images(j).name(1:end-4), '.png'));
        imwrite(patch, savedname, 'Mode', 'looless')
        fprintf('Image No. = %d\n', j);
    end
end

function dir = qmkdir(dir)
[success, message] = mkdir(dir);
end


