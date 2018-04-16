function ERGAS = ERGAS( X,Y,S )
%ERGAS 此处显示有关此函数的摘要
%   此处显示详细说明

% ig th downsamplig factor is not input, set it to S=4;
if nargin == 2
    S=32;
end

err_l2 = sqrt(mean(abs(Y-X).^2,2));

ERGAS = 100/(S)*sqrt(mean((err_l2./mean(X,2)).^2));

end

