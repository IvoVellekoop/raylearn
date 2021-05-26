function [col, row, mean_intensity, mean_masked_intensity, threshold, framemask, found] =...
    extract_pencil_position_from_frame(frame, meanthreshfactor, bgcornersize, percentile, percentilefactor, medfiltsize)
    % Extract Pencil Position Extract the column and row of the center of a pencil beam spot in an
    % image. A threshold is determined as the mean of the image multiplied by the meanthreshfactor
    % argument, or a factor times the percentile of the corner pixels, whichever is bigger. The
    % thresholded binary mask is then median filtered to remove remaining specks of noise. Then,
    % the mask is multiplied with the original image, setting the background to 0. The result is
    % sent to the img_center_of_mass function from the utilities repo.
    %
    % Input:
    % frame             2D numeric array containing the image.
    % meanthreshfactor  Positive scalar. This factor scales the threshold.
    % bgcornersize      Positive int. Size in pixels of corner square regions, which will be used
    %                   to determine background level.
    % percentile        Non-negative scalar. Percentile of corner pixel values to determine
    %                   background level.
    % percentilefactor  
    % medfiltsize       Positive integer. Size of the median filter.
    %
    % Output:
    % col, row          Column and row of the center of the pencil beam spot in the frame.
    % mean_intensity    Mean intensity of frame.
    % mean_masked_intensity  Mean intensity of masked frame.
    % threshold         Determined threshold for this frame.
    % framemask         Framemask for this frame, determined with threshold.
    % found             Logical. Whether a spot has been detected, i.e. any pixels above threshold.
    %
    % Requires: img_center_of_mass from the utilities repo.
    
    % Check input
    validateattributes(frame, {'numeric'}, {'2d'});
    validateattributes(meanthreshfactor, {'numeric'}, {'scalar', 'positive'});
    validateattributes(medfiltsize, {'numeric'}, {'scalar', 'positive', 'integer'});
    
    % Compute threshold and frame mask
    b = bgcornersize;
    cornerpixels = [frame(1:b, 1:b) frame(1:b, end-b+1:end)...          % Collect corner pixels
        frame(end-b+1:end, 1:b) frame(end-b+1:end, end-b+1:end)];
    threshold = max(meanthreshfactor * mean(frame, 'all'),...
        percentilefactor * prctile(cornerpixels, percentile, 'all'));
    framemask = medfilt2(frame > threshold, [medfiltsize medfiltsize]);
    masked_frame = framemask .* frame;
    
    % Determine intensity
    mean_intensity = mean(frame, 'all');
    mean_masked_intensity = mean(framemask, 'all');
    
    if any(framemask, 'all')        % Check if anything > threshold
        % Determine center of mass of masked image
        [col, row] = img_center_of_mass(masked_frame);
        found = true;
    else
        col = NaN;
        row = NaN;
        found = false;
    end
end






























