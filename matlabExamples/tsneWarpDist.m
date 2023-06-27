function warpDist = tsneWarpDist(d1, d2Mat, nTimeBinsPerTrial)
    %This implements the time-warp distance function from Willett et al.
    %2020.
    
    %d1 is a 1 x N vector representing a single data point that has been
    %'unrolled' from a matrix (T x D) into a vector (1 x TD), where T is
    %the number of time bins and D is the number of neural dimensions. This
    %'unrolled' representation is needed to work with MATLAB's tsne
    %function.
    
    %d2 is an M x N matrix, where each row is a data point.
    
    %nTimeBinsPerTrial specifies how many time bins (T) are included in each
    %data point. The number of neural dimensions is then D = N/T.
    
    %warpDist returns a M x 1 vector representing the distance between d1
    %and each row of d2.
    
    %affineWarps is a vector of alpha values to consider.
    affineWarps = linspace(0.7,1.42,15);
    
    %infer the number of neural dimensions per data point
    nNeuralDim = length(d1)/nTimeBinsPerTrial;
    
    %reshape d1 into a T x D matrix
    d1 = reshape(d1,nTimeBinsPerTrial,nNeuralDim);
    
    %eDist represnts the euclidean distance between d1 and all rows of d2
    %for each alpha. 
    eDist = zeros(size(d2Mat,1), length(affineWarps));
                
    %now we fill in eDist one entry at a time
    for a=1:length(affineWarps)
        %linearly warp d1 using this alpha
        d1_interp = interp1(1:size(d1,1), d1, linspace(1,size(d1,1),affineWarps(a)*size(d1,1)));

        %compute the euclidena distance between the warped d1 and all
        %points in d2
        for rowIdx=1:size(d2Mat,1)
            %reshape d2 into a T x D matrix
            d2 = d2Mat(rowIdx,:);
            d2 = reshape(d2,nTimeBinsPerTrial,nNeuralDim);

            %compute the euclidean distance, taking care to compute only 
            %over the relevant time points
            if affineWarps(a)>1
                df = d1_interp(1:size(d1,1),:)-d2;
            else
                df = d1_interp-d2(1:size(d1_interp,1),:);
            end
            eDist(rowIdx,a) = mean(df(:).^2);
        end
    end
    
    %the warp distance is defined as the minimum distance over all the
    %alphas, which we take here
    warpDist = min(eDist,[],2);
end