%This example script reproduces the t-sne plot from  fig. 1e and the
%nearest-neighbor classification accuracy results (94% accuracy).
%This was run and confirmed to produce correct results with MATLAB R2020a. 
%It takes ~15 minutes to run on my laptop. It's slow because we use a
%time-warp distance function that returns small distances for time series
%that are the same except for stretching/compression in time. This helps
%account for variability in writing speed. 

dat = load('C:\Users\wille\Documents\Data\Derived\handwritingDatasetsForRelease\RawData\t5.2019.05.08\singleLetters.mat');

letters = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',...
    'greaterThan','comma','apostrophe','tilde','questionMark'};

%%
%normalize the neural activity by blockwise z-scoring.  
for x=1:length(letters)
    normCube = single(dat.(['neuralActivityCube_' letters{x}]));
    
    %for each of the 9 blocks, subtract the block-specific mean and then
    %divide by the standard deviation.
    tIdx = 1:3;
    for y=1:9
        mn = zeros(3,1,192);
        mn(1,1,:) = dat.meansPerBlock(y,:);
        mn(2,1,:) = dat.meansPerBlock(y,:);
        mn(3,1,:) = dat.meansPerBlock(y,:);
        
        sd = zeros(1,1,192);
        sd(1,1,:) = dat.stdAcrossAllData;
        
        normCube(tIdx,:,:) = normCube(tIdx,:,:) - mn;
        normCube(tIdx,:,:) = normCube(tIdx,:,:) ./ sd;
        tIdx = tIdx + 3;
    end
    
    dat.(['neuralActivityCube_' letters{x}]) = normCube;
end

%%
%compute trial-averaged activity for each character, using a 50ms sd
%gaussian smoothing kernel. 
allData = zeros(2000,27264);
allSpatial = zeros(200000,192);
allLabels = zeros(2000,1);
allAvg = [];
cIdx = 1;
spatialIdx = 1;

for f=1:length(letters)
    letterCube = dat.(['neuralActivityCube_' letters{f}]);
    for x=1:size(letterCube,1)
        row = gaussSmooth_fast(squeeze(letterCube(x,60:end,:)), 3);

        row = row(:);
        allData(cIdx,:) = row;
        allLabels(cIdx,:) = f;
        cIdx = cIdx + 1;
        
        newChunk = gaussSmooth_fast(squeeze(letterCube(x,60:end,:)), 5);
        allSpatial(spatialIdx:(spatialIdx+size(newChunk,1)-1),:) = newChunk;
        spatialIdx = spatialIdx + size(newChunk,1);
    end
    
    avgLet = squeeze(mean(letterCube,1));
    avgLet = gaussSmooth_fast(avgLet(60:end,:), 5);
    allAvg = [allAvg; avgLet];
end

%%
%use the trial-averaged activity to compute PCs, then take the top 15 PCs
%as features for the single-trial data, using a 30 ms sd smoothing kernel.
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(allAvg);

nDim = 15;
allData = zeros(2000,142*nDim);
allLabels = zeros(2000,1);
cIdx = 1;
spatialIdx = 1;

for f=1:length(letters)
    tmp = dat.(['neuralActivityCube_' letters{f}]);
    for x=1:size(tmp,1)
        row = gaussSmooth_fast(squeeze(tmp(x,60:end,:)), 3);
        row = (row-MU)*COEFF(:,1:nDim);
         
        row = row(:);
        allData(cIdx,:) = row;
        allLabels(cIdx,:) = f;
        cIdx = cIdx + 1;
    end
end

allData(isnan(allData)) = 0;
allData = allData(1:(cIdx-1),:);
allLabels = allLabels(1:(cIdx-1));

%%
%tsne using warp-distance, this could take ~10-15 minutes. 
nTimeBinsPerPoint = 142;
warpDistFun = @(d1,d2)(tsneWarpDist(d1,d2,142));
[Y,loss] = tsne(allData,'Distance',warpDistFun,'Verbose',2,'Perplexity',40);

%%
%plot tsne results
plotLet = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',...
    '>',',','''','~','?'};

colors = [    0.3613    0.8000         0
    0.8000         0    0.1548
    0.8000    0.1548         0
    0.8000         0    0.4645
    0.6710         0    0.8000
    0.3613         0    0.8000
    0.5161    0.8000         0
    0.8000    0.6194         0
    0.8000         0         0
    0.6710    0.8000         0
    0.2065         0    0.8000
         0    0.1032    0.8000
    0.8000    0.3097         0
         0    0.7226    0.8000
         0    0.8000    0.5677
         0    0.8000    0.2581
         0    0.2581    0.8000
    0.0516         0    0.8000
    0.8000         0    0.6194
    0.8000    0.4645         0
    0.8000         0    0.7742
    0.8000    0.7742         0
         0    0.5677    0.8000
    0.8000         0    0.3097
         0    0.8000    0.4129
    0.0516    0.8000         0
         0    0.4129    0.8000
    0.5161         0    0.8000
    0.2065    0.8000         0
         0    0.8000    0.1032
         0    0.8000    0.7226];

ylims = [min(Y(:,2)), max(Y(:,2))];
xlims = [min(Y(:,1)), max(Y(:,2))];

figure('Color','w');
hold on;
for x=1:size(Y,1)
    text(Y(x,1), Y(x,2), plotLet{allLabels(x)},'Color',colors(allLabels(x),:),'FontWeight','bold','FontSize',6);
end
xlim(xlims);
ylim(ylims);
axis equal;
axis off;

%%
%k nearest neighbor classificaiton using warp-distance, this could take a few
%minutes to compute D. k=10
D = pdist(allData, warpDistFun);
D = squareform(D);

classAcc = zeros(size(D,1),1);
for x=1:length(classAcc)
   
    [~,sortIdx] = sort(D(x,:));
    sortIdx = sortIdx(2:11);
    choice = mode(allLabels(sortIdx));

    classAcc(x) = choice==allLabels(x);
end

disp('Warp NN Accuracy (CI)');
disp(mean(classAcc));
[PHAT, PCI] = binofit(sum(classAcc),length(classAcc));
disp(PCI);
