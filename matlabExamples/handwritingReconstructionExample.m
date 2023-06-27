%%
%This is a simplified example of how to reconstruct letter trajectories
%from neural activity (Fig. 1d) using 'templates' of what each letter should
%(generally) look like. Note that all reconstructions are held-out
%*letters*, so the reconstructions cannot overfit to the templates. 

%The timing of the templates won't match T5's writing style exactly. So
%first, we do an optimization over reaction time and template
%dilation/contraction to find the best overall parameters to start with
%that lead to the best linear velocity decoding performance. Then, we do a
%more specific optimization for each template individually to find the best
%parameters per-template. Finally, we do cross-validated letter
%reconstruction using those template parameters to train the decoder. 

%Cross-validation ensures that the reconstruction results are not a result
%of overfitting to the templates. If no consistent relationship between the
%neural activity and pen velocity exists, then decoding will not yield
%recognizable letter shapes for held-out letters.

%This is simplified code that doesn't match Fig. 1d exactly, but
%is close. Code could take 5-10 minutes to finish and was verified with Matlab R2020a. 

%%
%load time-warped neural data and the computer mouse letter templates
warpedCube = load('C:\Users\wille\Downloads\doi_10.5061_dryad.wh70rxwmv__v3 (2)\handwritingBCIData.tar\handwritingBCIData\handwritingBCIData\RNNTrainingSteps\Step1_TimeWarping\t5.2019.05.08_warpedCubes.mat');
mouseTemplates = load('C:\Users\wille\Downloads\doi_10.5061_dryad.wh70rxwmv__v3 (2)\handwritingBCIData.tar\handwritingBCIData\handwritingBCIData\Datasets\computerMouseTemplates.mat');

letters = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',...
    'greaterThan','comma','apostrophe','tilde','questionMark'};

%%
%fix the 'o' and 'x' templates which don't match T5's writing style on
%2019.05.08. He changed his writing style for certain letters on different
%days (which you can see by examining the reconstructions).

rot90 = [[cosd(-90), cosd(0)]; [sind(-90), sind(0)]];
mouseTemplates.o(:,1:2) = (rot90*mouseTemplates.o(:,1:2)')';

mouseTemplates.x(22:43,2) = -mouseTemplates.x(22:43,1);
mouseTemplates.x(47:68,1:2) = -mouseTemplates.x(47:68,1:2);

%%
%find initial reaction time and dilation parameters to globally apply to
%all templates based on whatever gives the best decoding performance
rtSteps = 0:2:20;
dilations = linspace(0.8,1.5,12);
corrHeatmap = zeros(length(rtSteps), length(dilations));
allFilts = cell(length(rtSteps), length(dilations));

for rIdx = 1:length(rtSteps)
    disp([' ' num2str(rIdx)]);
    for dIdx = 1:length(dilations)
        disp(['   ' num2str(dIdx)]);
        
        %unrolled data and templates
        unrolledNeural = [];
        unrolledTemplates = [];
        for letterIdx=1:length(letters)
            thisTemplate = mouseTemplates.(letters{letterIdx});
            thisTemplate = interp1(1:size(thisTemplate,1), thisTemplate, ...
                linspace(1, size(thisTemplate,1), size(thisTemplate,1)*dilations(dIdx)));
            thisTemplate = [zeros(rtSteps(rIdx),2); thisTemplate];
            thisTemplate = [thisTemplate; zeros(152-size(thisTemplate,1),2)];
            if size(thisTemplate,1)>152
                thisTemplate = thisTemplate(1:152,:);
            end
            newTemplate = repmat(thisTemplate, 27, 1);
            
            neuralCube = warpedCube.(letters{letterIdx});
            newAct = [];
            for t=1:size(neuralCube,1)
                newAct = [newAct; squeeze(neuralCube(t,50:end,:))];
            end
            
            unrolledNeural = [unrolledNeural; newAct];
            unrolledTemplates = [unrolledTemplates; newTemplate];
        end
        
        unrolledNeural(isnan(unrolledNeural)) = 0;
        
        %build filts
        filts = [ones(size(unrolledNeural,1),1), unrolledNeural]\unrolledTemplates;
        predVel = [ones(size(unrolledNeural,1),1), unrolledNeural]*filts;
        
        cMat = corr(predVel, unrolledTemplates);
        corrHeatmap(rIdx, dIdx) = mean(diag(cMat));
        allFilts{rIdx, dIdx} = filts;
    end
end

%find the best parameters and use the predicted velocity from those
%parameters
[~, maxIdx] = max(corrHeatmap(:));
[bestR, bestD] = ind2sub(size(corrHeatmap), maxIdx);
bestFilt = allFilts{bestR, bestD};

%%
%now optimize each template individually
possibleShifts = -5:10;
possibleDilations = linspace(0.8, 1.5, 10);
specificShifts = zeros(length(letters),1);
specificDilations = zeros(length(letters),1);

for letterIdx=1:length(letters)
    disp(letterIdx);
    thisHeatmap = zeros(length(possibleShifts), length(possibleDilations));
    for rIdx=1:length(possibleShifts)
        for dIdx=1:length(possibleDilations)
            thisTemplate = mouseTemplates.(letters{letterIdx});
            thisTemplate = interp1(1:size(thisTemplate,1), thisTemplate, ...
                linspace(1, size(thisTemplate,1), size(thisTemplate,1)*possibleDilations(dIdx)));
            thisTemplate = [zeros(rtSteps(bestR)+possibleShifts(rIdx),2); thisTemplate];
            thisTemplate = [thisTemplate; zeros(152-size(thisTemplate,1),2)];
            if size(thisTemplate,1)>152
                thisTemplate = thisTemplate(1:152,:);
            end
            newTemplate = repmat(thisTemplate, 27, 1);

            neuralCube = warpedCube.(letters{letterIdx});
            newAct = [];
            for t=1:size(neuralCube,1)
                newAct = [newAct; squeeze(neuralCube(t,50:end,:))];
            end

            newAct(isnan(newAct)) = 0;
            newVel = [ones(size(newAct,1),1), newAct] * bestFilt;
            thisHeatmap(rIdx, dIdx) = mean(diag(corr(newVel, newTemplate)));
        end
    end
    
    [~, maxIdx] = max(thisHeatmap(:));
    [bestSpecificR, bestSpecificD] = ind2sub(size(thisHeatmap), maxIdx);
    specificShifts(letterIdx) = rtSteps(bestR)+possibleShifts(bestSpecificR);
    specificDilations(letterIdx) = possibleDilations(bestSpecificD);
end

%%
%this variable defines which time step to stop drawing each letter (so that
%the reconstructions don't continue on past the point where T5 has finished
%drawing the letter).
manualEnds = [159 151 130 164 158 185 170 164 139 152 187 128 192 150 144 173 164 134 146 170 146 143 170 163 175 160 142 137 121 131 170];

%%
%Get cross-validated predictions for each letter (each letter is held-out
%completely). This ensures that the reconstruction results are not a result
%of overfitting to the templates. If no consistent relationship between the
%neural activity and pen velocity exists, then decoding will not yield
%recognizable letter shapes for held-out letters.
avgVel = cell(length(letters),1);
for outerLetterIdx=1:length(letters)
    disp(outerLetterIdx);
    
    %unrolled data and templates
    unrolledNeural = [];
    unrolledTemplates = [];
    for letterIdx=1:length(letters)
        %skip the testing letter
        if letterIdx==outerLetterIdx
            continue
        end
        
        thisTemplate = mouseTemplates.(letters{letterIdx});
        thisTemplate = interp1(1:size(thisTemplate,1), thisTemplate, ...
            linspace(1, size(thisTemplate,1), size(thisTemplate,1)*specificDilations(letterIdx)));
        thisTemplate = [zeros(specificShifts(letterIdx),2); thisTemplate];
        thisTemplate = [thisTemplate; zeros(152-size(thisTemplate,1),2)];
        if size(thisTemplate,1)>152
            thisTemplate = thisTemplate(1:152,:);
        end
        newTemplate = repmat(thisTemplate, 27, 1);

        neuralCube = warpedCube.(letters{letterIdx});
        newAct = [];
        for t=1:size(neuralCube,1)
            newAct = [newAct; squeeze(neuralCube(t,50:end,:))];
        end

        unrolledNeural = [unrolledNeural; newAct];
        unrolledTemplates = [unrolledTemplates; newTemplate];
    end

    unrolledNeural(isnan(unrolledNeural)) = 0;

    %build filts and apply to held out letter
    filts = [ones(size(unrolledNeural,1),1), unrolledNeural]\unrolledTemplates;
    neuralHeldOut = warpedCube.(letters{outerLetterIdx})(:,61:manualEnds(outerLetterIdx),:);
    neuralHeldOut = squeeze(nanmean(neuralHeldOut,1));
    neuralHeldOut = [ones(size(neuralHeldOut,1),1), neuralHeldOut] * filts;
    avgVel{outerLetterIdx} = neuralHeldOut;
end

%%
%plot letters
[cY, cX] = meshgrid(1:6,1:6);
cY = -cY;

figure('Position',[680 218 1024 880]);
hold on;

for c=1:26
    traj = cumsum(gaussSmooth_fast(avgVel{c},1.5));
    cn = [cX(c), cY(c)]*0.5;

    com = mean(traj);
    traj = traj - com;

    plot(cn(1) + traj(:,1), cn(2) + traj(:,2),'LineWidth', 2, 'Color', [0    0.4470    0.7410]);
    plot(cn(1) + traj(1,1), cn(2) + traj(1,2), 'o', 'Color', [0.8500    0.3250    0.0980]);
    text(cn(1), cn(2)+0.3, letters{c},'FontSize',18,'FontWeight','bold');
end

axis equal;
axis off;