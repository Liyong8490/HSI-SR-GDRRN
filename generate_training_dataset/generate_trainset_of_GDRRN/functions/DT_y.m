function   z    =   DT_y(y, sf, sz)
[ch,n]                   =    size(y);
s0                       =    floor(sf/2);
if  ch  == 1
    z                        =    zeros(sz);    
    z(s0:sf:end, s0:sf:end)  =    reshape(y, floor(sz/sf));
    z                        =    z(:)';
else    
    z                        =    zeros(ch, n*sf^2);
    t                        =    zeros(sz);
    for  i  = 1 : ch
        t(s0:sf:end,s0:sf:end)        =    reshape(y(i,:), floor(sz/sf));
        z(i,:)                        =    t(:)';
    end
end

