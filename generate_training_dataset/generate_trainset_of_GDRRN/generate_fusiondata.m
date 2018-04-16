clc;
clear;

addpath('functions/');
datasetname = 'Harvard';
datasetpath = ['../', datasetname, '/'];

files = dir(datasetpath);
files = files(3:end);

patch_size = 32;
strides = 16;
scale_up = 4;
scales = [1.0, 0.5, 0.25];
flage = 0;
step = 2;
maxfile = ceil(length(files)/step);
%% count
count = 0;
load([datasetpath,files(1).name]);
SF = 1;
[bands, lines, samples] = size(data);
data = reshape(reshape(data, bands, lines*samples)',lines, samples, bands);
data = data/(1.0*max(max(max(data))));
num = zeros(3,1);
for s=1:length(scales)
    GT              =   imresize(data, scales(s));
    [lines_, samples_, bands] = size(GT);
    sz              =   [lines_, samples_];
    par             =   Parameters_setting( scale_up, 'Gaussian_blur', sz );
    GT_t            =   reshape(GT, lines_*samples_, bands)';
    GT_t            =   reshape(GT_t, bands, lines_, samples_);
    [hsi_] = Im2Patch_bicubic_single(GT_t, patch_size, strides);
    num(s) = size(hsi_, 1);
    count = count + num(s)*8;
end
par.P           =   create_P();
count = count*maxfile
%% generate
batchsize = 32;
Totalnum = ceil(count/batchsize)*batchsize;
% data_f = zeros(Totalnum, bands*2, patch_size, patch_size, 'single');
c = zeros(Totalnum, size(par.P,1), patch_size, patch_size, 'single');
hsi = zeros(Totalnum, bands, patch_size/scale_up, patch_size/scale_up, 'single');
% hsi_t = zeros(Totalnum, bands, patch_size, patch_size, 'single');
label = zeros(Totalnum, bands, patch_size, patch_size, 'single');
for i = 1:step:length(files)
    i
    load([datasetpath,files(i).name]);
    SF = 1;
    [bands, lines, samples] = size(data);
    data = reshape(reshape(data, bands, lines*samples)',lines, samples, bands);
    data = data/(1.0*max(max(max(data))));
    for s=1:length(scales)
        GT              =   imresize(data, scales(s));
        [lines_, samples_, bands] = size(GT);
        sz              =   [lines_, samples_];
        for mode = 1:8
            par             =   Parameters_setting( scale_up, 'Gaussian_blur', sz );
            num_t           =   num*8;
            GT_t            =   augmentation(GT, mode);
            GT_t            =   reshape(GT_t, lines_*samples_, bands)';
            H               =   par.H(GT_t);
            H               =   reshape(H,bands,lines_/scale_up, samples_/scale_up);
            
%             H_t               =   par.H(GT_t)';
%             H_t               =   reshape(H_t,lines_/scale_up, samples_/scale_up,bands);
%             H_t               =   imresize(H_t, scale_up);
%             H_t               =   reshape(H_t, lines_*samples_, bands)';
%             H_t               =   reshape(H_t, bands, lines_, samples_);
            
            par.P           =   create_P();
            M               =   par.P*GT_t;
            M               =   reshape(M, size(par.P, 1), lines_, samples_);
            GT_t            =   reshape(GT_t, bands, lines_, samples_);
            
            [hsi_, c_, label_] = Im2Patch(GT_t, M, H, patch_size, strides, scale_up);
            % [hsi_, c_, label_, hsi_t_] = Im2Patch_trpl(GT_t, M, H, H_t, patch_size, strides, scale_up);
            % [c_, label_, hsi_t_] = Im2Patch_trpl_bicubic(GT_t, M, H, H_t, patch_size, strides, scale_up);
            
            if s == 1
                ssnum = 0;
            elseif s == 2
                ssnum = num_t(1);
            elseif s == 3
                ssnum = num_t(1) + num_t(2);
            end
            axiss = sum(num_t)*floor(i/step) + ssnum + num(s)*(mode-1);
            hsi( axiss+1 : axiss+num(s), :, :, :) = hsi_;
            c( axiss+1 : axiss+num(s), :, :, :) = c_;
            label( axiss+1 : axiss+num(s), :, :, :) = label_;
            % hsi_t( axiss+1 : axiss+num(s), :, :, :) = hsi_t_;
            
        end
    end

end
size(label)
%%
save_path = ['train_data/','fusion_trainset_',datasetname,'_x',num2str(scale_up),'_', num2str(patch_size), '/'];
mkdir(save_path);
filename_gt = ['gt', '.h5'];
filename_c = ['c', '.h5'];
filename_hsi = ['hsi','.h5'];
% filename_hsi_t = ['hsi_t','.h5'];

hsi = permute(hsi,[4,3,2,1]);
c = permute(c,[4,3,2,1]);
label = permute(label,[4,3,2,1]);
% hsi_t = permute(hsi_t,[4,3,2,1]);
%%
tic
for iter = 1:size(label, 4)
    % hsi
    hsi_in = hsi(:,:,:,iter);
    h5create([save_path,filename_hsi], ['/',num2str(iter)], size(hsi_in), 'Datatype','single'); %,'ChunkSize',size(hsi_in),'Deflate',3);
    h5write([save_path,filename_hsi], ['/',num2str(iter)], hsi_in);
    % c
    c_in = c(:,:,:,iter);
    h5create([save_path,filename_c], ['/',num2str(iter)], size(c_in), 'Datatype','single'); %,'ChunkSize',size(c_in),'Deflate',3);
    h5write([save_path,filename_c], ['/',num2str(iter)], c_in);
    % label
    gt_in = label(:,:,:,iter);
    h5create([save_path,filename_gt], ['/',num2str(iter)], size(gt_in), 'Datatype','single'); %,'ChunkSize',size(gt_in),'Deflate',3);
    h5write([save_path,filename_gt], ['/',num2str(iter)], gt_in);
    % hsi_t
%     hsi_t_in = hsi_t(:,:,:,iter);
%     h5create([save_path,filename_hsi_t], ['/',num2str(iter)], size(hsi_t_in), 'Datatype','single'); %,'ChunkSize',size(gt_in),'Deflate',3);
%     h5write([save_path,filename_hsi_t], ['/',num2str(iter)], hsi_t_in);
    if mod(iter,1000) == 1
        iter, toc
    end
end
iter, toc

