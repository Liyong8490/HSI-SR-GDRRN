function   z    =   BT_y(y, fft_BT, sz)
[ch,n]                   =    size(y);
if  ch  == 1
    z                        =    reshape(y, sz);
    z                        =    real( ifft2(fft2( z ).*fft_BT) );
    z                        =    z(:)';
else    
    z                        =    zeros(ch, n);
    for  i  = 1 : ch
        t                    =    reshape(y(i,:), sz);
        Htz                  =    real( ifft2(fft2( t ).*fft_BT) );
        z(i,:)               =    Htz(:)';
    end
end

