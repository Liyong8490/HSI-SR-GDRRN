function  [c, label, hsi_t]  =  Im2Patch_trpl_bicubic( GT,M,H, H_t,win, strides, SF )
%win_h = win/SF;
%strides_h = strides/SF;

patch = GT(:, 1:strides:end-win+1, 1:strides:end-win+1);
TotalPatNum =   size(patch,2)*size(patch,3);                                %Total Patch Number in the image
label           =   zeros(size(GT,1), TotalPatNum, win*win, 'single');       %Current Patches
%hsi           =   zeros(size(H,1), TotalPatNum, win_h*win_h, 'single');
c           =   zeros(size(M,1), TotalPatNum, win*win, 'single');
hsi_t           =   zeros(size(H_t,1), TotalPatNum, win*win, 'single');

k = 0;
for i  = 1:win
    for j  = 1:win
        k = k + 1;
        E_patch         =  GT(:, i:strides:end-win+i,j:strides:end-win+j);      
        label(:, :, k)      =  reshape(E_patch, size(GT,1), TotalPatNum, 1);
        M_patch         =  M(:, i:strides:end-win+i,j:strides:end-win+j);      
        c(:, :, k)      =  reshape(M_patch, size(M,1), TotalPatNum, 1);
        H_t_patch         =  H_t(:, i:strides:end-win+i,j:strides:end-win+j);      
        hsi_t(:, :, k)      =  reshape(H_t_patch, size(H_t,1), TotalPatNum, 1);
    end
end
% k = 0;
% for i  = 1:win_h
%     for j  = 1:win_h
%         k = k + 1;
%         H_patch         =  H(:, i:strides_h:end-win_h+i,j:strides_h:end-win_h+j);      
%         hsi(:, :, k)      =  reshape(H_patch, size(H,1), TotalPatNum, 1);
%     end
% end
label = permute(label, [2,1,3]);
label = reshape(label, TotalPatNum, size(GT,1), win, win);
c = permute(c, [2,1,3]);
c = reshape(c, TotalPatNum, size(M,1), win, win);
%hsi = permute(hsi, [2,1,3]);
%hsi = reshape(hsi, TotalPatNum, size(H,1), win_h, win_h);
hsi_t = permute(hsi_t, [2,1,3]);
hsi_t = reshape(hsi_t, TotalPatNum, size(H_t,1), win, win);

