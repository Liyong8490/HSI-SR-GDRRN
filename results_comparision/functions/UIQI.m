function mean_Q = UIQI(X,Y)

mX = mean(X,2);
varX = var(X,0,2);
mY = mean(Y,2);
varY = var(Y,0,2);

covXY =  mean(X.*Y,2) - mX.*mY;

Q = 4*covXY.*mX.*mY./(varX + varY)./(mX.^2 + mY.^2);
mean_Q=mean(Q);

end

