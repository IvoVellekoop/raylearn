function [col, row, mean_intensity, mean_masked_intensity, threshold, framemask, found, num_pixels_at_edge] =...
    extract_pencil_position_from_frame(frame, percentile, percentilefactor, min_threshold, min_mask_size_pix, medfiltsize, erodestrel)
    % Extract Pencil Position Extract the column and row of the center of a pencil beam spot in an
    % image. A threshold is determined as the mean of the image multiplied by the meanthreshfactor
    % argument, or a factor times the percentile of the corner pixels, whichever is bigger. The
    % thresholded binary mask is then median filtered to remove remaining specks of noise. Then,
    % the mask is multiplied with the original image, setting the background to 0. The result is
    % sent to the img_center_of_mass function from the utilities repo.
    %
    % Input:
    % frame             2D numeric array containing the image.
    % percentile        Non-negative scalar. Percentile of pixel values to determine threshold.
    % percentilefactor  Factor to multiply with percentile to determine threshold.
    % medfiltsize       Positive integer. Size of the median filter.
    % erodestrel        Structering element for erode operation.
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
    validateattributes(medfiltsize, {'numeric'}, {'scalar', 'positive', 'integer'});
    
    % Compute threshold and frame mask
    threshold_prctile = percentilefactor * prctile(frame, percentile, 'all');   % Compute threshold
    threshold = max(threshold_prctile, min_threshold);
    framemask_medfilt = medfilt2(frame > threshold, [medfiltsize medfiltsize]); % Median filter
    framemask = imerode(framemask_medfilt, erodestrel);
    masked_frame = framemask .* frame;
    
    % Determine intensity
    mean_intensity = mean(frame, 'all');
    mean_masked_intensity = mean(framemask, 'all');
    
    num_pixels_at_edge = sum(framemask(1, :)) + sum(framemask(:, 1)) ...
        + sum(framemask(end, :)) + sum(framemask(:, end));
    
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

