clear;
clc;

addpath('functions/');
datasetname = 'Harvard';
upscale = 8;
datapath = ['../testset/',num2str(upscale),'/',datasetname,'/'];
files = dir(datapath);
files = files(3:end);
aver_PSNR = 0;
aver_SAM = 0;
aver_UIQI = 0;
aver_ERGAS = 0;
outpath = ['bilinear_results/',datasetname,'/',num2str(upscale),'/'];
mkdir(outpath);
for i = 1:length(files)
    % bicubic
    load([datapath, files(i).name]);
    if strcmp(datasetname, 'Harvard')
        GT = GT(:,1:end/2,1:end/2);
        H = H(:,1:end/2,1:end/2);
        M = M(:,1:end/2,1:end/2);
    end
    [bands, lines, samples] = size(GT);
    upsample_t = reshape(reshape(H, size(H,1), size(H,2)*size(H,3))', size(H,2), size(H,3), size(H,1));
    upsample_t = imresize(upsample_t, upscale, 'bilinear');
    upsample = reshape(upsample_t, size(H,2)*size(H,3)*upscale*upscale, size(H,1))';
    GT = reshape(GT, size(GT,1), size(GT,2)*size(GT,3));
    n_PSNR = PSNR(upsample, GT);
    n_SAM = SAM(upsample, GT);
    n_UIQI = UIQI(upsample, GT);
    n_ERGAS = ERGAS(upsample, GT);
    aver_PSNR = aver_PSNR + n_PSNR;
    aver_SAM = aver_SAM + n_SAM;
    aver_UIQI = aver_UIQI + n_UIQI;
    aver_ERGAS = aver_ERGAS + n_ERGAS;
    result = reshape(upsample,size(H,1), size(H,2)*upscale, size(H,3)*upscale);
    save([outpath,files(i).name(1:end-4),'_recon.mat'], 'result');
    fprintf(['\tBilinear\t',files(i).name,'\tPSNR = %f\tSAM = %f\tUIQI = %f\tERGAS = %f\n'],n_PSNR, n_SAM, n_UIQI, n_ERGAS);
end
fprintf('Bilinear\tAverPSNR = %f\tAverSAM = %f\tAverUIQI = %f\tAverERGAS = %f\n',aver_PSNR/length(files),aver_SAM/length(files),aver_UIQI/length(files),aver_ERGAS/length(files));
