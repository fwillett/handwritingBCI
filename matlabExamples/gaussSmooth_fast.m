function [ Y ] = gaussSmooth_fast( timeSeries, width )
    if width==0
        Y = timeSeries;
        return;
    end
    
    wingSize = ceil(width*5);
    gKernel = normpdf( -wingSize:wingSize, 0, width);
    normFactor = cumsum(gKernel)';
    
    test = [timeSeries; zeros(length(gKernel)-1,size(timeSeries,2))];
    Y = filter(gKernel, 1, test);
    Y(1:(length(gKernel)-1),:) = bsxfun(@times, Y(1:(length(gKernel)-1),:), 1./normFactor(1:(end-1)));
    Y((end-length(gKernel)+2):end,:) = bsxfun(@times, Y((end-length(gKernel)+2):end,:), flipud(1./normFactor(1:(end-1))));
    Y = Y((1+(length(gKernel)-1)/2):(end-(length(gKernel)-1)/2),:);
end

