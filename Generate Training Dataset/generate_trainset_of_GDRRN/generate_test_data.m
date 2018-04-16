clc;
clear;

addpath('functions/');
datasetname = 'Harvard';
datasetpath = ['../', datasetname, '/'];

files = dir(datasetpath);
files = files(3:end);

scale_up = 4;
out_path = ['../testset/',num2str(scale_up),'/',datasetname, '/'];
mkdir(out_path);
flage = 0;
for i = 1:length(files)
    load([datasetpath,files(i).name]);
    [bands, lines, samples] = size(data);
    data = reshape(reshape(data, bands, lines*samples)',lines, samples, bands);
    data = data/(1.0*max(max(max(data))));
    GT              =   imresize(data, 1.0);
    [lines_, samples_, bands] = size(GT);
    sz              =   [lines_, samples_];
    GT              =   reshape(GT, lines_*samples_, bands)';
    par             =   Parameters_setting( scale_up, 'Gaussian_blur', sz );
    H               =   par.H(GT);
    H               =   reshape(H, bands, lines_/scale_up, samples_/scale_up);
    par.P           =   create_P();
    M               =   par.P*GT;
    M               =   reshape(M, size(par.P, 1), lines_, samples_);
    GT              =   reshape(GT, bands, lines_, samples_);
%     GT = permute(GT,[3,2,1]);
%     H = permute(H,[3,2,1]);
%     M = permute(M,[3,2,1]);

    save([out_path,files(i).name], 'H','M','GT');
end

