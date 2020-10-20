function bg = bg_grating(bgtype, ppp, pmin, pmax, bg_pix)
    % Compute the background grating SLM pixel values
    %
    % Input:
    % bgtype:   Choose between 'sine', 'blaze' or 'step'
    % ppp:      1D Array of positive scalars. Array of pixels per period.
    % pmin:     Nonnegative integer. Pixel value of minimum.
    % pmax:     Nonnegative integer. Pixel value of maximum.
    % bg_pix:   Positive integer. Number of pixels for background image.
    %
    % Output:
    % bg:       1D Array of positive scalars. The computed grating pattern.
    
    yslm = 0:bg_pix-1;            % Array of pixel number
    
    % Calculate background grating SLM values
    switch bgtype
        case 'sine'
            bg = pmin + (pmax-pmin) * (0.5 + 0.5 * cos(yslm*2*pi / ppp));
        case 'blaze'
            bg = pmin + (pmax-pmin) * mod(yslm / ppp, 1);
        case 'step'
            bg = pmin + (pmax-pmin) * (mod(yslm / ppp, 1) < 0.5);
        otherwise
            error('Specify a grating by setting the variable bgtype')
    end
end