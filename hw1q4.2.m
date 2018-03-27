%Generating data
n = 250;
d = 80;
K = 10;
sigma = 1;
b = 0;
W_orig = zeros(d,1);
W_orig(1:K,1) = W_orig(1:K, 1) + 10;
W = zeros(d,1);
e = normrnd(0, sigma^2, n, 1);
X = normrnd(0, sigma^2, d, n);
Y = X'*W_orig + e;

epsilon = 0.1;
residual = zeros(n,1);
lambda = 2*max(X*(Y - mean(Y)));
lambda_list = zeros(10,1);
precision = zeros(10,1);
recall = zeros(10,1);

A = zeros(d,1);
C = zeros(d,1);
A = 2 * sum(X.^2, 2);


for i = 1:10
    loss_old = 0;
    delta_loss = intmax('int64');
    while delta_loss > epsilon
        residual = Y - X'*W - b;
        b_old = b;
        b = mean(Y - X'*W - residual);
        residual = residual + b_old - b;
        
        for k = 1:d
            C(k) = 2*X(k,:)*residual + 2*X(k,:)*X(k,:)'*W(k);
            w_old = W;
            
            if C(k) < -lambda
                W(k) = (C(k) + lambda)/A(k);
            elseif C(k) > lambda
                W(k) = (C(k) - lambda)/A(k); 
            else
                W(k) = 0;
            end
            
            residual = residual + X'*(w_old - W);
        end
        loss_new = sum(residual.^2) + lambda*(sumabs(W));
        delta_loss = abs(loss_old - loss_new);
        loss_old = loss_new;
    end
    lambda_list(i) = lambda;
    lambda = lambda/2;
    precision(i) = nnz(W(1:K))/nnz(W);
    recall(i) = nnz(W(1:K))/K;
end

figure
plot(lambda_list, precision, lambda_list, recall);
title('Precision/Recall Plot');
xlabel('Lambda');
ylabel('Precision/Recall');
ylim([0 1.5]);
legend('Precision','Recall','Location','northeast');


