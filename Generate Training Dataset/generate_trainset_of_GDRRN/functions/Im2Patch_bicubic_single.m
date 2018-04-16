function  [data_]  =  Im2Patch_bicubic_single( GT, win, strides )
%win_h = win/SF;
%strides_h = strides/SF;

patch           =   GT(:, 1:strides:end-win+1, 1:strides:end-win+1);
TotalPatNum     =   size(patch,2)*size(patch,3);                                %Total Patch Number in the image
data_           =   zeros(size(GT,1), TotalPatNum, win*win, 'single');       %Current Patches

k = 0;
for i  = 1:win
    for j  = 1:win
        k = k + 1;
        E_patch             =  GT(:, i:strides:end-win+i,j:strides:end-win+j);      
        data_(:, :, k)      =  reshape(E_patch, size(GT,1), TotalPatNum, 1);
    end
end
data_ = permute(data_, [2,1,3]);
data_ = reshape(data_, TotalPatNum, size(GT,1), win, win);

