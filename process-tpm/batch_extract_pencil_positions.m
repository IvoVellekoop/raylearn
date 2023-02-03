% Batch Extract Pencil Positions
% Walk through (split) data files and extract the pencil beam position from each frame. Then
% save the results.

forcereset = 0;

if forcereset || ~exist('dirs', 'var')
    close all; clear; clc

    % Define location paths
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
    dirconfig_raylearn
end

% Settings
dosavedata = 1;                     % Toggle saving data
doshowplot = 0;                     % Toggle showing plot of each found position (for debugging)
dosaveplot = 0;                     % Toggle saving plots
plotsubdir = 'plot';                % Subdirectory for saving plot frames
fig_size_pix = 700;                 % Figure size in pixels

percentile = 99.7;                  % Percentile of the frame values to use
percentilefactor = 0.70;            % Multiply percentile pixel value with this
medfiltsize = 5;                    % Size of the median filter
erodestrel = strel('disk', 4);      % Erode filter structering element
max_num_pixels_at_edge = 4;         % Maximum number of mask pixels at the edge before a measurement is considered as failed
inputpattern = [dirs.localdata '/raylearn-data/TPM/pencil-beam-raw/*/raylearn_pencil_beam*'];
% inputpattern = [dirs.localdata '\raylearn-data\TPM\pencil-beams-split\04-May-2021-grid\raylearn_pencil_beam*'];
outputgroupfolder = [dirs.localdata '/raylearn-data/TPM/pencil-beam-positions'];

% Process
assert(exist(dirs.localdata, 'dir'), 'Directory "%s" doesn''t exist', dirs.localdata)
dirlist = dir(inputpattern);        % List files to be processed
D = length(dirlist);                % Number of measurements
assert(D > 0, 'Directory list is empty. Please check input pattern.')

% Compute total size of files
bytestotal = 0;
for d = 1:D
    % List files in directory (corresponding to same measurement)
    filelistpattern = fullfile(dirlist(d).folder, dirlist(d).name, 'raylearn_pencil_beam*');
    filelist = dir(filelistpattern);
    F = length(filelist);
    for f = 1:F
        bytestotal = bytestotal + filelist(f).bytes;
    end
end
fprintf('\nTotal data size: %s\nLoading data...\n', bytes2str(bytestotal))

if doshowplot
    fig = figure;
    fig_resize(fig_size_pix, 1.2, 0, fig);
    movegui('center');
end

% Loop over directories
% for d = 1:D
starttime = now;
bytesdone = 0;
for d = 1:D
    total_found = 0;
    num_processed = 0;

    % List files in directory (corresponding to same measurement)
    filelistpattern = fullfile(dirlist(d).folder, dirlist(d).name, 'raylearn_pencil_beam*');
    filelist = dir(filelistpattern);
    F = length(filelist);
    
    % Loop over files
    for f = 1:F
        filepath = fullfile(filelist(f).folder, filelist(f).name);      % Construct file path
        load(filepath)
        
        [~,~,G] = size(frames_ft);
        
        if f==1
            % Initialize list of camera coordinates and intensities
            % Note: this code is inside the for loop as it relies on the loaded data
            cam_ft_col  = nan(F, G);
            cam_ft_row  = nan(F, G);
            cam_ft_mean_intensity = nan(F, G);
            cam_ft_mean_masked_intensity = nan(F, G);
            cam_ft_mask_area = nan(F, G);
            
            cam_img_col = nan(F, G);
            cam_img_row = nan(F, G);
            cam_img_mean_intensity = nan(F, G);
            cam_img_mean_masked_intensity = nan(F, G);
            cam_img_mask_area = nan(F, G);
            found_spot = false(F, G);

            frame_ft_sum = zeros(size(frames_ft(:,:,1)));
            frame_img_sum = zeros(size(frames_img(:,:,1)));
            
            if doshowplot && dosaveplot
                mkdir(fullfile(filelist(f).folder, plotsubdir))
            end
        end

        % Sum of all frames
        frame_ft_sum = frame_ft_sum + sum(frames_ft, 3);
        frame_img_sum = frame_img_sum + sum(frames_img, 3);

        
        % Loop over frames
        for g=1:G
            rawframe_ft = frames_ft(:,:,g);
            frame_ft = rawframe_ft - single(darkframe_ft);        % Substract darkframe
            frame_ft(frame_ft < 0) = 0;                           % Remove < noise
            
            rawframe_img = frames_img(:,:,g);
            frame_img = rawframe_img - single(darkframe_img);     % Substract darkframe
            frame_img(frame_img < 0) = 0;                         % Remove < noise

            % Extract pencil beam spot position
            [col_ft, row_ft, mean_intensity_ft, mean_masked_intensity_ft, threshold_ft, framemask_ft, found_ft, num_pixels_at_edge_ft] = extract_pencil_position_from_frame(...
                frame_ft, percentile, percentilefactor, medfiltsize, erodestrel);
            
            [col_img, row_img, mean_intensity_img, mean_masked_intensity_img, threshold_img, framemask_img, found_img, num_pixels_at_edge_img] = extract_pencil_position_from_frame(...
                frame_img, percentile, percentilefactor, medfiltsize, erodestrel);
            
            % Count failed detections
            total_found = total_found + found_ft*found_img;
            
            % Show plot of found pencil beam positions on frame
            if doshowplot
                figure(fig);
                
                % Plot original ft frame with position
                subplot(2,2,1)
                plotframe(frame_ft, row_ft, col_ft,...
                    sprintf('Fourier Cam\ndir: %i/%i | col: %.1f, row: %.1f', d, D, col_ft, row_ft))
                
                % Plot ft mask with position
                subplot(2,2,2)
                plotframe(framemask_ft, row_ft, col_ft,...
                    sprintf('Fourier Mask\nthreshold: %.1f', threshold_ft));
                
                % Plot original img frame with position
                subplot(2,2,3)
                plotframe(frame_img, row_img, col_img,...
                    sprintf('Image Cam\ndir: %i/%i | col: %.1f, row: %.1f', d, D, col_ft, row_img))
                
                % Plot img mask with position
                subplot(2,2,4)
                plotframe(framemask_img, row_img, col_img,...
                    sprintf('Image Mask\nthreshold: %.1f', threshold_img));
                
                drawnow
                
                if dosaveplot       % Save figures as PNG images
                    figure_path = fullfile(filelist(f).folder, plotsubdir, sprintf('frame_f%i-g%i.png', f, g));
                    fprintf('Saving figure f%i,g%i to: %s\n', f, g, figure_path)
                    saveas(fig, figure_path)
                end
            end
            
            % Store found beam spot positions and intensities
            cam_ft_col(s, g) = col_ft;
            cam_ft_row(s, g) = row_ft;
            cam_ft_mean_intensity(s, g) = mean_intensity_ft;
            cam_ft_mean_masked_intensity(s, g) = mean_masked_intensity_ft;
            cam_ft_mask_area(s, g) = sum(framemask_ft, [1 2]);
            
            cam_img_col(s,g) = col_img;
            cam_img_row(s,g) = row_img;
            cam_img_mean_intensity(s, g) = mean_intensity_img;
            cam_img_mean_masked_intensity(s, g) = mean_masked_intensity_img;
            cam_img_mask_area(s, g) = sum(framemask_ft, [1 2]);
            
            found_spot(s, g) = found_ft & found_img ...
                & (num_pixels_at_edge_ft  < max_num_pixels_at_edge) ...
                & (num_pixels_at_edge_img < max_num_pixels_at_edge);
            
            num_processed = num_processed + 1;
        end
        
        
        bytesdone = bytesdone + filelist(f).bytes;          % Count size of processed files
        not_found = num_processed - total_found;            % Number of failed spot detections
        
        eta(bytesdone, bytestotal, starttime, 'console',...
            sprintf(['Extracting Pencil Positions...\nDirectories done: %i/%i'...
                     '\nFiles size done: %s/%s'...
                     '\n%s\nFile: %i/%i\n\nFailed detections: %i/%i (%.0f%%)'],...
            d, D, bytes2str(bytesdone), bytes2str(bytestotal),...
            dirlist(d).name, f, F, not_found, num_processed, 100*not_found/num_processed), 0);
    end
    
    
    % Save found beam spot positions
    if dosavedata
        disp('Saving file...')
        
        % Construct save path
        [~, subdir] = fileparts(dirlist(d).folder);
        savedir = fullfile(outputgroupfolder, subdir);
        warning('off', 'MATLAB:MKDIR:DirectoryExists');
        try mkdir(savedir); catch 'MATLAB:MKDIR:DirectoryExists'; end   % Create savedir if needed
        savepath = fullfile(savedir, [dirlist(d).name '.mat']);
        
        % Save
        save(savepath, '-v7.3', 'copt_ft', 'copt_img', 'sopt', 'p',...
            'cam_ft_col', 'cam_ft_row', 'cam_ft_mean_intensity', 'cam_ft_mean_masked_intensity',...
            'cam_img_col', 'cam_img_row', 'cam_img_mean_intensity', 'cam_img_mean_masked_intensity',...
            'found_spot', 'found_ft', 'found_img', 'num_pixels_at_edge_ft', 'num_pixels_at_edge_img')
    end
end



function plotframe(frame, row, col, titlestr)
    % Plot camera frame or framemask along with found beam spot coordinate
    %
    % Input:
    % frame     2D numeric array. Frame(mask) to plot.
    % row       Numeric. Row coordinate of beam spot.
    % col       Numeric. Column coordinate of beam spot.
    % titlestr  String/char array. Title above plot.
    
    % Compute axis ranges and coords
    pix_size_mm = 5.5e-3;
    xsize_mm = size(frame, 2) * pix_size_mm;
    ysize_mm = size(frame, 1) * pix_size_mm;
    xdata = [-xsize_mm xsize_mm]/2;
    ydata = [-ysize_mm ysize_mm]/2;
    
    x = pix_size_mm * (row - size(frame, 2)/2);
    y = pix_size_mm * (col - size(frame, 1)/2);
    
    % Plot
    imagesc(xdata, ydata, frame);
    colorbar;
    axis image
    hold on
    plot(x, y, '+k')
    hold off
    
    % Text
    title(titlestr)
    set(gca, 'fontsize', 14)
    xlabel('x (mm)')
    ylabel('y (mm)')
end

