function [result, model] = rtd(A, I, B, C, D, E, X, Y, Z, k, nIter)
% This code gives a regularized tensor and matrix decomposition algorithm.
% 
% Input: Tensor A (m*n*r): user-location-activity tensor, incomplete
%        Indicator tensor I (m*n*r): the entries used as training data is 0, otherwise 1
%        Matrix B (m*m): user-user social network
%        Matrix C (n*p): location-feature
%        Matrix D (r*r): activity-activity correlation
%        Matrix E (m*n): user-location visiting history
%        Matrix X (m*k): initial values for low-dimensional representation for the users
%        Matrix Y (n*k): initial values for low-dimensional representation for the locations
%        Matrix Z (r*k): initial values for low-dimensional representation for the activities
%        Integer k: the number of low dimensions, 1 <= k <= min(m,n,r)
%        Integer nIter: the number of iterations for gradient descent
% 
% Output: Structure result: result.loss keeps the objective function values, 
%                           result.err keeps the prediction errors measured by RMSE (root mean square error)
%         Structure model: model.X, model.Y and model.Z are the output low-dimensional representations for the users, locations and activities respectively, 
%                          model.P is the output low-dimensional representations for the location features,
%                          model.U is the output low-dimensional representations
%
% Note: This code uses one auxilliary toolbox.
%       (1) Tensor toolbox (matlab interface): version 2.4.
%           See http://csmr.ca.sandia.gov/~tgkolda/TensorToolbox/.
%
% For more details about this code, please refer to our paper:
%       Collaborative Filtering Meets Mobile Recommendation: A User-centered Approach.
%       Vincent W. Zheng, Bin Cao, Yu Zheng, Xing Xie and Qiang Yang. 
%       In Proc. of the 24th AAAI Conference on Artificial Intelligence (AAAI-10). Atlanta, Georgia, USA. July 11-15, 2010.
%
% Copyright by the paper authors.
% Any question, please send email to vincentz_AT_cse.ust.hk.
% May 20th, 2010.
% ===============================================================

a = 0.1; % user-user
b = 0.1; % loc-fea
c = 0.1; % act-act
d = 0.1; % user-loc
e = 0.1; % regularization

% step size
alpha = 0.0001;

[m, n] = size(E);
p = size(D, 1);
q = size(C, 2);

% split the training data and test data
IndTrain = I;
IndTest = ~I;
ATest = A.*IndTest;
A = A.*IndTrain;

% require the auxilliary tensor toolbox
A1 = tenmat(A,1);
A2 = tenmat(A,2);
A3 = tenmat(A,3);

A1 = A1.data;
A2 = A2.data;
A3 = A3.data;

I = eye(k);

% get the laplacian matrices
LB = diag(sum(B)) - B;
LD = diag(sum(D)) - D;

U = rand(q,k);

V{1} = X;
V{2} = Y;
V{3} = Z;

% construct the tensor based on the initial values for X, Y and Z
P = ktensor(V);
P = tensor(P);

oldLoss = 100000;

for iIter=1:nIter
    
    dX = - A1*khatrirao(Z,Y) + X*((Z'*Z).*(Y'*Y) + e * I) + a*LB*X + d*(X*Y'-E)*Y;
    dY = - A2*khatrirao(Z,X) + Y*((Z'*Z).*(X'*X) + e * I) + b*(Y*U'-C)*U + d*(X*Y'-E)'*X;
    dZ = - A3*khatrirao(Y,X) + Z*((Y'*Y).*(X'*X) + e * I) + c*LD*Z;
    dU = b*(Y*U'-C)'*Y + e*U;

    X = X - alpha*dX;
    Y = Y - alpha*dY;
    Z = Z - alpha*dZ;
    U = U - alpha*dU;

    V{1} = X;
    V{2} = Y;
    V{3} = Z;
    
    P = ktensor(V);  
    P = tensor(P);

    loss(iIter) = norm(A - P)^2 + e*(norm(X,'fro')^2 + norm(Y,'fro')^2 + norm(Z,'fro')^2 + norm(U,'fro')^2) ...
        + a/2*trace(X'*LB*X) + b/2*norm(C - Y*U', 'fro')^2 + c/2*trace(Z'*LD*Z) + d/2*norm(E-X*Y','fro')^2;
    
    % the error is measured by RMSE
    err(iIter) = norm(ATest - IndTest.*P);
    fprintf('Iteration: %d\t loss: %f\t error: %f\n', iIter, loss(iIter), err(iIter));
    
    if(loss(iIter) < oldLoss)
        oldLoss = loss(iIter);
    else
        X = X + alpha*dX;
        Y = Y + alpha*dY;
        Z = Z + alpha*dZ;
        U = U + alpha*dU;
        V{1} = X;
        V{2} = Y;
        V{3} = Z;
        P = ktensor(V);  
        P = tensor(P);
        break;
    end
end

% return the model and result
model.P = P;
model.X = X;
model.Y = Y;
model.Z = Z;
model.U = U;
result.loss = loss;
result.err = sqrt(norm(ATest - IndTest.*P)^2/nnz(ATest));
