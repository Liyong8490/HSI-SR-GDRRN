clear;
clc;

addpath('functions/');
datasetname = 'Harvard';
mkdir('results/');
GDRRNpath = ['../results/'];
resultspathes = dir(GDRRNpath);
resultspathes = resultspathes(3:end);
for method_c = 1:length(resultspathes)
    methodname = resultspathes(method_c).name;
    fprintf(['Start method %d/%d: ',methodname, '***********************\n'], method_c, length(resultspathes));
    savename = ['results/',methodname,'.mat'];
    if exist(savename, 'file')
        continue
    end
    % methodname = ['results_HSI_SR_GDRRN_Harvard_up',num2str(upscale),'_','saml_1e1_g1'];
    
    num_models = dir([GDRRNpath, methodname, '/', datasetname,'/']);
    upscale = str2num(num_models(3).name);
    num_models = dir([GDRRNpath, methodname, '/', datasetname,'/',num2str(upscale),'/']);
    num_models = num_models(3:end);
    num_Epochs = length(num_models);
    if num_Epochs < 30
        continue
    end
    datapath = ['../testset/',num2str(upscale),'/Harvard/'];
    files = dir(datapath);
    files = files(3:end);
    names = cell(length(files),1);
    for i=1:length(files)
        names(i) = cellstr(files(i).name);
    end
    all_measures = zeros(num_Epochs, length(files)+1, 4);
    
    for i = 1:length(files)
        % bicubic
        load([datapath, files(i).name]);
        GT = GT(:,1:end/2,1:end/2);
        H = H(:,1:end/2,1:end/2);
        M = M(:,1:end/2,1:end/2);
        [bands, lines, samples] = size(GT);
        GT = reshape(GT, size(GT,1), size(GT,2)*size(GT,3));
        % GDRRN
        for iteration=1:num_Epochs
            fullpath = [GDRRNpath, methodname, '/', datasetname,'/',num2str(upscale),'/', num2str(iteration), '/'];
            load([fullpath, files(i).name(1:end-4), '_recon.mat']);
            result = reshape(result, size(result,1), size(result,2)*size(result,3));
            n_GDRRN_PSNR = PSNR(result, GT);
            n_GDRRN_SAM = SAM(result, GT);
            n_GDRRN_UIQI = UIQI(result, GT);
            n_GDRRN_ERGAS = ERGAS(result, GT);
            data = [n_GDRRN_PSNR, n_GDRRN_SAM, n_GDRRN_UIQI, n_GDRRN_ERGAS];
            all_measures(iteration, i, :) = data(:);
            % fprintf(['\tGDRRN\t',files(i).name,'\tPSNR = %f\tSAM = %f\tUIQI = %f\tERGAS = %f\n'],n_GDRRN_PSNR, n_GDRRN_SAM, n_GDRRN_UIQI, n_GDRRN_ERGAS);
        end
        fprintf([methodname, '\n!!!!!*****\t%d\t*****!!!!!\n'], i);
    end
    for j=1:num_Epochs
        for k = 1:4
            all_measures(j,length(files) + 1, k) = sum(all_measures(j, :, k))./length(files);
        end
    end
    
    save(savename, 'names', 'all_measures');
end
