function [col, row, threshold, framemask] =...
    extract_pencil_position_from_frame(frame, meanthreshfactor, bgcornersize, percentile, percentilefactor, medfiltsize)
    % Extract Pencil Position
    % Extract the column and row of the center of a pencil beam spot in an image. A threshold
    % is determined as the mean of the image multiplied by the meanthreshfactor argument. The
    % thresholded binary mask is then median filtered to remove remaining specks of noise.
    % Then, the mask is multiplied with the original image, setting the background to 0. The
    % result is sent to the img_center_of_mass function from the utilities repo.
    % %%%%
    %
    % Input:
    % frame             2D numeric array containing the image.
    % meanthreshfactor  Positive scalar. This factor scales the threshold.
    % medfiltsize       Positive integer. Size of the median filter.
    %
    % Output:
    % Column and row of the center of the pencil beam spot in the frame.
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
    
    if any(framemask, 'all')        % Check if anything > threshold
        [col, row] = img_center_of_mass(framemask .* frame);
    else
        warning('Could not find beam spot above threshold.')
        col = NaN;
        row = NaN;
    end
end
