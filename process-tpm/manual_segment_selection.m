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
fig_size_pix = 800;                 % Figure size in pixels

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
            
            frames_ft_maxG = zeros([size(frames_ft(:,:,1)), F]);
            frames_img_maxG = zeros([size(frames_img(:,:,1)), F]);
            
            if doshowplot && dosaveplot
                mkdir(fullfile(filelist(f).folder, plotsubdir))
            end
        end
        
        % Loop over frames
        for g=1:G
            rawframe_ft = frames_ft(:,:,g);
            frame_ft = rawframe_ft - single(darkframe_ft);        % Substract darkframe
            frame_ft(frame_ft < 0) = 0;                           % Remove < noise
            
            rawframe_img = frames_img(:,:,g);
            frame_img = rawframe_img - single(darkframe_img);     % Substract darkframe
            frame_img(frame_img < 0) = 0;                         % Remove < noise
            
            frames_ft_maxG(:, :, f) = max(frames_ft_maxG(:, :, f), frame_ft);
            frames_img_maxG(:, :, f) = max(frames_img_maxG(:, :, f), frame_img);
        end
        
        bytesdone = bytesdone + filelist(f).bytes;          % Count size of processed files
        not_found = num_processed - total_found;            % Number of failed spot detections
        
        eta(bytesdone, bytestotal, starttime, 'console',...
            sprintf(['Extracting Pencil Positions...\nDirectories done: %i/%i'...
                     '\nFiles size done: %s/%s'...
                     '\n%s\nFile: %i/%i\n'],...
            d, D, bytes2str(bytesdone), bytes2str(bytestotal),...
            dirlist(d).name, f, F), 0);
    end
    
end
