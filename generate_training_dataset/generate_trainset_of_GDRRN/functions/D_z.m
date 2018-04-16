function   x    =   D_z(z, sf, sz)
[ch, n]         =    size(z);
s0              =    floor(sf/2);
if ch==1
    Hz          =    reshape(z, sz);
    x           =    Hz(1:sf:end, 1:sf:end);
    x           =    (x(:))';
else
    x           =    zeros(ch, floor(n/(sf^2)));    
    for  i  = 1 : ch
        Hz         =    reshape(z(i,:), sz);
        t          =    Hz(s0:sf:end, s0:sf:end);
        x(i,:)     =    (t(:))';
    end
end
