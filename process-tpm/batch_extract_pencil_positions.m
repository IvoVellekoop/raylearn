% Batch Extract Pencil Positions
% Walk through (split) data files and extract the pencil beam position from each frame. Then
% save the results.

close all; clear; clc

% Define location paths
addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
dirconfig_raylearn

% Settings
doshowplot = 0;                     % Toggle showing plot of each found position (for debugging)
dosave = 1;                         % Toggle saving
meanthreshfactor = 4;               % This factor scales the threshold
bgcornersize = 10;                  % Corner size in pixels. These will be used for noise level estimation.
percentile = 97;                    % Percentile of the corner pixel values to use
percentilefactor = 2;               % 
medfiltsize = 5;                    % Size of the median filter
inputpattern = 'F:\ScientificData\raylearn-data\TPM\pencil-beams-split\*\raylearn_pencil_beam*';
outputgroupfolder = 'F:\ScientificData\raylearn-data\TPM\pencil-beam-positions';

% Process
dirlist = dir(inputpattern);        % List files to be processed
D = length(dirlist);

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

if doshowplot
    fig = figure;
    fig_resize(900, 1.2, 0, fig);
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
        
        [~,~,~,G] = size(frames_slmslice_ft);
        
        if f==1
            % Initialize list of camera coordinates
            % Note: The number of 
            cam_ft_col  = nan(F, G);
            cam_ft_row  = nan(F, G);
            cam_img_col = nan(F, G);
            cam_img_row = nan(F, G);
            found_spot = false(F, G);
        end

        
        % Loop over frames
        for g=1:G
            frame_ft = frames_slmslice_ft(:,:,1,g);
            frame_img = frames_slmslice_img(:,:,1,g);
            
            % Extract pencil beam spot position
            [col_ft, row_ft, threshold_ft, framemask_ft, found_ft] = extract_pencil_position_from_frame(...
                frame_ft, meanthreshfactor, bgcornersize, percentile, percentilefactor, medfiltsize);
            
            [col_img, row_img, threshold_img, framemask_img, found_img] = extract_pencil_position_from_frame(...
                frame_img, meanthreshfactor, bgcornersize, percentile, percentilefactor, medfiltsize);
            
            % Count failed detections
            total_found = total_found + found_ft*found_img;
            
            % Show plot of found pencil beam positions on frame
            if doshowplot
                figure(fig);
                
                % Plot original ft frame with position
                subplot(2,2,1)
                plotframe(frame_ft, row_ft, col_ft,...
                    sprintf('Frame\ndir: %i/%i | col: %.1f, row: %.1f', d, D, col_ft, row_ft))
                
                % Plot ft mask with position
                subplot(2,2,2)
                plotframe(framemask_ft, row_ft, col_ft,...
                    sprintf('Mask\nthreshold: %.1f', threshold_ft));
                
                % Plot original img frame with position
                subplot(2,2,3)
                plotframe(frame_img, row_img, col_img,...
                    sprintf('Frame\ndir: %i/%i | col: %.1f, row: %.1f', d, D, col_ft, row_img))
                
                % Plot img mask with position
                subplot(2,2,4)
                plotframe(framemask_img, row_img, col_img,...
                    sprintf('Mask\nthreshold: %.1f', threshold_img));
                
                drawnow
            end
            
            % Store found beam spot positions
            cam_ft_col(s, g) = col_ft;
            cam_ft_row(s, g) = row_ft;
            cam_img_col(s,g) = col_img;
            cam_img_row(s,g) = row_img;
            found_spot(s, g) = found_ft & found_img;
            
            num_processed = num_processed + 1;
        end
        
        
        bytesdone = bytesdone + filelist(f).bytes;          % Count size of processed files
        not_found = num_processed - total_found;            % Number of failed spot detections
        
        eta(bytesdone, bytestotal, starttime, 'console',...
            sprintf(['Extracting Pencil Positions...\nDirectories done: %i/%i'...
                     '\n%s\nFile: %i/%i\n\nFailed detections: %i/%i (%.0f%%)'],...
            d, D, dirlist(d).name, f, F, not_found, num_processed, 100*not_found/num_processed), 0);
    end
    
    
    % Save found beam spot positions
    if dosave
        disp('Saving file...')
        
        % Construct save path
        [~, subdir] = fileparts(dirlist(d).folder);
        savedir = fullfile(outputgroupfolder, subdir);
        warning('off', 'MATLAB:MKDIR:DirectoryExists');
        try mkdir(savedir); catch 'MATLAB:MKDIR:DirectoryExists'; end   % Create savedir if needed
        savepath = fullfile(savedir, [dirlist(d).name '.mat']);
        
        % Save
        save(savepath, '-v7.3', 'copt_ft', 'copt_img', 'sopt', 'p',...
            'cam_ft_col', 'cam_ft_row', 'cam_img_col', 'cam_img_row', 'found_spot')
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
    
    imagesc(frame);
    colorbar;
    axis image
    hold on
    plot(row, col, '+k')
    hold off
    title(titlestr)
end

