function   x    =   B_z(z, fft_B, sz)
[ch, n]         =    size(z);
if ch==1
    Hz          =    real( ifft2(fft2( reshape(z, sz) ).*fft_B) );
    x           =    (Hz(:))';
else
    x           =    zeros(ch, n);    
    for  i  = 1 : ch
        Hz         =    real( ifft2(fft2( reshape(z(i,:), sz) ).*fft_B) );
        x(i,:)     =    (Hz(:))';
    end
end


