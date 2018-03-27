D = load('trainData.txt');
X = sparse(D(:,2), D(:,1), D(:,3));
[d, n] = size(X);
Y = load('trainLabels.txt');

V = load('valData.txt');
val_X = sparse(V(:,2), V(:, 1), V(:, 3));
val_Y = load('valLabels.txt');
T = load('testData.txt');
test_X = sparse(T(:,2), T(:, 1), T(:, 3));
features  = strings;
iLine = 1;
fid1 = fopen('featureTypes.txt', 'r');
while ~feof(fid1)
    line = fgetl(fid1);
    features = [features, line];
    line = line+1;
end
features = features(1, 2:d+1);

A = zeros(d,1);
C = zeros(d,1);

A = 2 * sum(X.^2, 2);
lambda = 2*max(X*(Y - mean(Y)));
rmse_val = zeros(1,10);
rmse_train = zeros(1, 10);
diff = -1;

idx = 1;
b = 0;
W = zeros(d,1);
lam = zeros(10,1);
non_zeros = zeros(1,10);

while diff < 0
    delta_loss = intmax('int64');
    loss_old = 0;
    iter = 1;
    while delta_loss > 0.1
        residual = Y - X'*W - b;
        b_old = b;
        b = mean(residual + b);
        residual = residual + b_old - b;
        
        for k = 1:d
            C(k) = 2*X(k,:)*residual + 2*X(k,:)*X(k,:)'*W(k);
            w_old = W;
            
            if C(k) < -lambda
                W(k) = (C(k) + lambda)/A(k);
            elseif C(k) >= -lambda && C(k) <= lambda
                W(k) = 0;
            else
                W(k) = (C(k) - lambda)/A(k); 
            end
            
            residual = residual + X'*(w_old - W);
        end
        loss_new = sum(residual.^2) + lambda*(sum(abs(W)));
        delta_loss = abs(loss_old - loss_new);
        loss_old = loss_new;
    end
%     sqrt(mean((Y - X'*W - b).^2))
    lam(idx) = lambda;
    rmse_val(1, idx) = sqrt(sum((val_Y - val_X'*W - b).^2)/n);
    lambda = lambda/2;
    if idx == 1
        diff = intmin('int64');
    else
        diff = rmse_val(1, idx) - rmse_val(1, idx-1);
    end
    rmse_train(1, idx) = sqrt(sum((Y - X'*W - b).^2)/n);
    non_zeros(1, idx) = nnz(W);
    idx = idx + 1;
end 


rmse_val
rmse_train
non_zeros

plot(lam, rmse_val, lam, rmse_train, '--');
title('RMSE Validation/RMSE train');
xlabel('Lambda');
ylabel('RMSE Validation/RMSE train');
ylim([1 3]);
legend('RMSE Validation','RMSE train','Location','northeast');

pred_Y = test_X'*W + b;
csvwrite('predTestLabels.csv', pred_Y);

[Weights, I] = sort(W,'descend');


for i = d-10:d
   features(1, I(i));
end

figure
plot(lam, non_zeros);
title('Non Zeros vs Lambda');
xlabel('Lambda');
ylabel('Non Zeros');
ylim([1 1200]);
legend('Non_zeros','Location','northeast');




