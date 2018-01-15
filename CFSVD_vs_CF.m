%% Fresh Opening
clear all; close all; clc;


%% Load Dataset

% Load .mat Matrix 
load('jesterdata1.mat')
%Ratings are real values ranging from -10.00 to +10.00 (the value "99" corresponds to "null" = "not rated").
%One row per user
%The first column gives the number of jokes rated by that user. The next 100 columns give the ratings for jokes 01 - 100.
%The sub-matrix including only columns {5, 7, 8, 13, 15, 16, 17, 18, 19, 20} is dense. Almost all users have rated those jokes (see discussion of "universal queries" in the above paper).

%% Data Pre-processing
% Create a pure joke dataRawRating.mat matrix slide out the first colum
dataRawRating = jesterdata1(:,2:end);

% Create a rating times for each joke dataNumRating.mat matrix slide out the first colum
dataNumRating = jesterdata1(:,1);


%% CP vs CP-SVD Training Set and Testing Set

% Train and test on the dense colum
% Seprate the dense data out
denseSetidx = find(dataNumRating == 100); 
denseSet = zeros(length(denseSetidx),100); %7200*100

for i = denseSetidx
    denseSet = dataRawRating(i,:);
end 
% sperate training and testing set form the denseSet randomly.

%% Set different ratio testing / total
for x = 0.2:0.1:0.9
ratio = x; % testing / total
trainingSet = denseSet;
msize = numel(trainingSet);
testingSetEntriesNum = round(ratio*msize);
trainingSet(randperm(msize,testingSetEntriesNum )) = 99;

% Prediction Generation for the Training Set
% Create a pure joke dataRawRating.mat matrix slide out the first colum
dataRawRating_ = trainingSet;

dataNumRating = trainingSet(:,1); % get the same length 
% Create a rating times for each joke dataNumRating.mat matrix slide out the first colum
for i = 1:length(trainingSet(:,1))
    dataNumRating(i) = length(find(trainingSet(i,:) ~= 99));
end 

% use average rating for a Joke to replace the Null aka 99 to capture
% meaningful latent relationship

% Inital a vector of size dataNumRating for rating average for each joke
ratingAverage = dataNumRating; % Just to get the same length colum, not for geting the data

ratingAverageJoke = zeros(1,100);
% Make all null rating joke to zero for calcualtion the average rating of
dataRawZeroRating = dataRawRating_;
dataRawZeroRating(dataRawZeroRating == 99) = 0;

% Calculate the average rating into ratingAverage
for idx_cos = 1:7200
    ratingAverage(idx_cos) = sum(dataRawZeroRating(idx_cos,:))/dataNumRating(idx_cos);
end

% Calculate the average rating into ratingAverageJoke
for idx_joke = 1:100
    ratingAverageJoke(idx_joke) = sum(dataRawZeroRating(:,idx_joke))/length(find(dataRawRating(:,idx_joke) ~= 99));
end

% compute R_norm
rNorm = dataRawRating_;


for idx_joke = 1:100
    temp = rNorm(idx_joke,:);
    temp(temp == 99) = ratingAverageJoke(idx_joke);
    rNorm(idx_joke,:) = temp;
end

ratingFilled = rNorm;

for idx_cos = 1:7200
    temp = rNorm(idx_cos,:) - ratingAverage(idx_cos);
    rNorm(idx_cos,:) = temp;
end



% factor R_nrom use SVD
[U,S,V] = svd(rNorm,'econ');

%% Save NMAE with regard value K
k = 20;
    
Sk = S(1:k,1:k);
Vk = V(1:k,:);
Uk = U(:,1:k);

% Compute resultant matrices UkSk SkVk
SkSR = sqrt(Sk);
UkSk = Uk * SkSR;
SkVk = SkSR * Vk;

% % prediction recommendation score for any customer c and product p 
% C = 1; % customer
% P = 1; % product
% tempA = UkSk(C,:);
% tempB = SkVk(:,P);
% Cpredict = dot(tempA,tempB) + ratingAverage(C);

% prediction recommendation score for all customer c and product p matrix
dataPredRating = dataRawRating_;
for C = 1:7200 % customer
    for P = 1:100 % product
        tempA = UkSk(C,:);
        tempB = SkVk(:,P);
        Cpredict = dot(tempA,tempB) + ratingAverage(C);
        dataPredRating(C,P) = Cpredict;
    end
end

%% Optimize the predicted result
% Reduce MAE by setting predicted rating higher than 10 as 10 and
% lower than -10 as -10
dataPredRating(dataPredRating > 10) = 10;
dataPredRating(dataPredRating < -10) = -10;


%% MEAN-ABSOLUTE-ERROR
% % First calculate the "error" part.
% err = Actual - Predicted;
% % Then take the "absolute" value of the "error".
% absoluteErr = abs(err);
% % Finally take the "mean" of the "absoluteErr".
% meanAbsoluteErr = mean(absoluteErr)

% You can just use the built in Mean Absolute Error function and pass in
% the "error" part.
% MeanAbsoluteError?MAE?
MAE = mae(denseSet-dataPredRating);
% Normalized MAE independent from rating scale
NMAE = MAE/20;

filename = sprintf('CFSVD_%s_K_%d_X_%d.mat','NMAE',k,x);
save(filename,'NMAE')


end


i = 0;
Y_CFSVD = 0.2:0.1:0.9;
for x = 0.2:0.1:0.9
filename = sprintf('CFSVD_%s_K_%d_X_%d.mat','NMAE',20,x);
load(filename,'NMAE')
i = i +1;
Y_CFSVD(i) = NMAE;
end


%% Set different ratio testing / total for CF
for x = 0.2:0.1:0.9
ratio = x; % testing / total
trainingSet = denseSet;
msize = numel(trainingSet);
testingSetEntriesNum = round(ratio*msize);
trainingSet(randperm(msize,testingSetEntriesNum )) = 99;

% Prediction Generation for the Training Set
% Create a pure joke dataRawRating.mat matrix slide out the first colum
dataRawRating_ = trainingSet;

dataNumRating = trainingSet(:,1); % get the same length 
% Create a rating times for each joke dataNumRating.mat matrix slide out the first colum
for i = 1:length(trainingSet(:,1))
    dataNumRating(i) = length(find(trainingSet(i,:) ~= 99));
end 

% use average rating for a Joke to replace the Null aka 99 to capture
% meaningful latent relationship

% Inital a vector of size dataNumRating for rating average for each joke
ratingAverage = dataNumRating; % Just to get the same length colum, not for geting the data

ratingAverageJoke = zeros(1,100);
% Make all null rating joke to zero for calcualtion the average rating of
dataRawZeroRating = dataRawRating_;
dataRawZeroRating(dataRawZeroRating == 99) = 0;

% Calculate the average rating into ratingAverage
for idx_cos = 1:7200
    ratingAverage(idx_cos) = sum(dataRawZeroRating(idx_cos,:))/dataNumRating(idx_cos);
end

% Calculate the average rating into ratingAverageJoke
for idx_joke = 1:100
    ratingAverageJoke(idx_joke) = sum(dataRawZeroRating(:,idx_joke))/length(find(dataRawRating(:,idx_joke) ~= 99));
end

% compute R_norm
rNorm = dataRawRating_;


for idx_joke = 1:100
    temp = rNorm(idx_joke,:);
    temp(temp == 99) = ratingAverageJoke(idx_joke);
    rNorm(idx_joke,:) = temp;
end

ratingFilled = rNorm;

for idx_cos = 1:7200
    temp = rNorm(idx_cos,:) - ratingAverage(idx_cos);
    rNorm(idx_cos,:) = temp;
end


%% CFpure
dataPredRating = rNorm;


%% Optimize the predicted result
% Reduce MAE by setting predicted rating higher than 10 as 10 and
% lower than -10 as -10
dataPredRating(dataPredRating > 10) = 10;
dataPredRating(dataPredRating < -10) = -10;


%% MEAN-ABSOLUTE-ERROR
% % First calculate the "error" part.
% err = Actual - Predicted;
% % Then take the "absolute" value of the "error".
% absoluteErr = abs(err);
% % Finally take the "mean" of the "absoluteErr".
% meanAbsoluteErr = mean(absoluteErr)

% You can just use the built in Mean Absolute Error function and pass in
% the "error" part.
% MeanAbsoluteError?MAE?
MAE = mae(denseSet-dataPredRating);
% Normalized MAE independent from rating scale
NMAE = MAE/20;

filename = sprintf('CF_%s_X_%d.mat','NMAE',x);
save(filename,'NMAE')


end


i = 0;
Y_CF = 0.2:0.1:0.9;
for x = 0.2:0.1:0.9
filename = sprintf('CF_%s_X_%d.mat','NMAE',x);
load(filename,'NMAE')
i = i +1;
Y_CF(i) = NMAE;
end

%% Plot CF_SVD vs pureCF
Y_CFSVD = fliplr(Y_CFSVD);
Y_CF = fliplr(Y_CF);

x = 0.1:0.1:0.8;
plot(x,Y_CFSVD,'-o',x,Y_CF,'-o')
ylabel('Normalized MAE')
xlabel('x train test ratio')
title('Plot SVD vs PureCF prediction (k is fixed at 20)')
legend('CF-SVD','PureCF')














