function [PSNR] = PSNR(recon,truth)
% RMSE in 8-bit range [0...255]

diff = double(im2uint8(truth))-double(im2uint8(recon));

MSE = sum(sum(diff.^2)) /size(truth,1) /size(truth,2);

RMSE = sqrt( MSE );

% These are equivalent defintions
%
% RMSE = sqrt( norm(diff,'fro')^2 /size(truth,1) /size(truth,2) );
% 
% RMSE =  norm(diff,'fro') * sqrt(1 /size(truth,1) /size(truth,2) );
PSNR = 10*log10(255^2/RMSE^2);

