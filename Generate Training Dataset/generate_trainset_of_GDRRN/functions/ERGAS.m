function ERGAS = ERGAS( X,Y,S )
%ERGAS �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

% ig th downsamplig factor is not input, set it to S=4;
if nargin == 2
    S=32;
end

err_l2 = sqrt(mean(abs(Y-X).^2,2));

ERGAS = 100/(S)*sqrt(mean((err_l2./mean(X,2)).^2));

end

