%% Measure pencil beams
% Use The Galvo Mirrors and the SLM to create pencil beams
% with various angles and record the outgoing pencil beams.

%% Reset variables and hardware connections
doreset = 0;

if doreset || ~exist('slm', 'var') || ~exist('daqs', 'var')  || ~exist('cam_ft', 'var')  || ~exist('cam_img', 'var')
    close all; clear; clc
    setup_raylearn_exptpm
end

%% Settings
p.samplename = '400um_aligned_to_slm';
doshowcams = 0;                     % Toggle show what the cameras see
dosave = 1;                         % Toggle savings
dochecksamplename = 0;              % Toggle console sample name check

% SLM Settings
p.segment_patch_id = 2;             % Pencil Beam segment SLM patch ID
p.ppp = 2;                          % Pixels per period for the grating. Should match Galvo setting!
p.segmentsize_pix = 50 * p.ppp;     % Segment width in pixels
p.beamdiameter = 0.50;              % Diameter of circular SLM segment set (relative coords)
p.slm_offset_x = 0.00;              % Horizontal offset of rectangle SLM geometry (relative coords)
p.slm_offset_y = 0.00;              % Vertical offset of rectangle SLM geometry (relative coords)
p.N_diameter =  7;                  % Number of segments across SLM diameter

% Galvo Mirror settings
p.GalvoNX =  5;                     % Number of Galvo steps, x
p.GalvoNY =  5;                     % Number of Galvo steps, y
p.GalvoXcenter = -0.558;            % Galvo center x
p.GalvoYcenter =  0.050;            % Galvo center y
p.GalvoRadius  =  0.030;            % Galvo scan radius: from center to outer
% Note: the actual number of galvo steps is smaller, as the corners from the square grid
% will be cut to make a circle

% Ask if samplename is correct
if dochecksamplename
    fprintf('\nSample name set to <strong>%s</strong>. Is this correct?\nPress Enter to continue...\n', p.samplename)
    pause
end


%% Compute constants
% Compute segment size in relative coords

[p.NxSLM, p.NySLM] = size(slm.getPixels);                   % Retrieve SLM size in pixels
% p.NySLM = 500;  %%% Use only for debugging, when SLM is unavailable
p.segmentwidth = p.segmentsize_pix / p.NySLM;               % Segment width in relative coords
p.segmentheight = p.segmentwidth;                           % Segment height in relative coords

% Create blaze grated segment, masked with circle
xblaze = linspace(-1,1,p.segmentsize_pix);                  % Create x coords for segment pixels
yblaze = xblaze';                                           % Create x coords for segment pixels
blazeradius = (p.segmentsize_pix+1) / p.segmentsize_pix;    % Radius from center to borderpixel edge
mask_outside_circle = (xblaze.^2+yblaze.^2) > blazeradius;  % Mask for pixels outside circle
p.blaze = single(bg_grating('blaze', p.ppp, 0, 255, p.segmentsize_pix) .* ones(p.segmentsize_pix,1));
p.blaze(mask_outside_circle) = 0;                           % Set pixels outside circle to 0

% Create set of SLM segment rectangles within circle
[p.rects, S, p.inside_circle_mask, p.segments_X, p.segments_Y] = ...
    BlockedCircleSegments(p.N_diameter, p.beamdiameter, p.slm_offset_x, p.slm_offset_y, p.segmentwidth, p.segmentheight);

% Create set of Galvo tilts
% (Make a grid that's 5% smaller than the defined radius, to make it fit)
p.galvoXs1d = linspace(p.GalvoXcenter-p.GalvoRadius*0.95, p.GalvoXcenter+p.GalvoRadius*0.95, p.GalvoNX);
p.galvoYs1d = linspace(p.GalvoYcenter-p.GalvoRadius*0.95, p.GalvoYcenter+p.GalvoRadius*0.95, p.GalvoNY);
[galvoXsq, galvoYsq] = meshgrid(p.galvoXs1d, p.galvoYs1d);
% Create circular mask to pick grid points
p.galvo_scan_mask = ((galvoXsq-p.GalvoXcenter).^2 + (galvoYsq-p.GalvoYcenter).^2 <= p.GalvoRadius.^2);
p.galvoXs = single(galvoXsq(p.galvo_scan_mask));
p.galvoYs = single(galvoYsq(p.galvo_scan_mask));
G = numel(p.galvoXs);

% 
% p.galvoXs = p.GalvoXcenter; %%%%%%%%%%%%
% p.galvoYs = p.GalvoYcenter;
% G = 1;
% %%%%%%%%%%%%%%%%%%

frame_ft_sum  = zeros(copt_ft.Width,  copt_ft.Height);
frame_img_sum = zeros(copt_img.Width, copt_img.Height);

%% Capture dark frames
outputSingleScan(daqs, [p.GalvoXcenter, p.GalvoYcenter]);   % Set Galvo
slm.setData(p.segment_patch_id, 0);                         % Set SLM segment pixels to 0
slm.update;

cam_ft.trigger;
darkframe_ft = cam_ft.getData;                              % Fourier Plane camera dark frame
cam_img.trigger;
darkframe_img = cam_img.getData;                            % Image Plane camera dark frame


%% Main measurement loop
slm.setData(p.segment_patch_id, p.blaze);

if doshowcams                       % Spawn wide figure
    figure; plot(0,0);
    fig_resize(500, 2.5);
end

if dosave
    % Script path, revision and date&time
    p.script_name = mfilename('fullpath');
    [~, p.script_version] = system(['git --git-dir="' dirs.repo '\.git" rev-parse HEAD']);
    [~, p.git_diff] = system(['git --git-dir="' dirs.repo '\.git" diff']);
    p.save_time = now;

    % Create save directory
    filenameprefix = 'raylearn_pencil_beam';
    p.savename = sprintf('%s_%f_%s', filenameprefix, now, p.samplename);
    p.subdir = ['\raylearn-data\TPM\pencil-beam-raw\' date '-' p.samplename];
    p.savedir = fullfile([dirs.localdata p.subdir], p.savename);
    try mkdir(p.savedir); catch 'MATLAB:MKDIR:DirectoryExists'; end
    fprintf('\nSave Directory: %s\n', p.savedir)
end

starttime = now;
SG = S*G;
sg = 0;
rawdatasize = bytes2str(4 * SG * (copt_ft.Height*copt_ft.Width + copt_img.Height*copt_img.Width));
fprintf('\nRaw data size will be: %s\n', rawdatasize)
disp('Started measurement...')
for s = 1:S                        % Loop over SLM segments
    numchars = 0;
    
    %%%% I want to switch to 4D indexing at some point. When I implement camera/galvo triggering
    %%%% for fast scanning might be a good time.
    
    % Initialize arrays
    frames_ft  = zeros(copt_ft.Width,  copt_ft.Height,  G, 'single');
    frames_img = zeros(copt_img.Width, copt_img.Height, G, 'single');

    for g = 1:G                     % Loop over Galvo tilts
        sg = sg+1;                  % Counter for ETA
        outputSingleScan(daqs, [p.galvoXs(g), p.galvoYs(g)]);   % Set Galvo Mirror
        pause(0.002)
    
        % Incoming angle of pencil beam: set SLM segment
        slm.setRect(p.segment_patch_id, p.rects(s,:));
        slm.update

        % Capture camera frames
        cam_ft.trigger;
        frame_ft = single(cam_ft.getData);
        frame_ft_sum = frame_ft_sum + frame_ft;
        frames_ft(:,:,g) = frame_ft;

        cam_img.trigger;
        frame_img = single(cam_img.getData);
        frame_img_sum = frame_img_sum + frame_img;
        frames_img(:,:,g) = frame_img;

        % Show what's on the cameras and check for overexposure
        if doshowcams
            subplot(1,2,2)
            imagesc(frame_ft)
            saturation_ft = 100 * max(frame_ft,[],'all') / 65520;   % 12bit -> 2^16 - 2^4 = 65520
            title(sprintf('Fourier Plane Cam | Segment %i/%i\nsaturation = %.0f%%', s, S, saturation_ft))
            colorbar

            subplot(1,2,1)
            imagesc(frame_img)
            saturation_img = 100 * max(frame_img,[],'all') / 65520; % 12bit -> 2^16 - 2^4 = 65520
            title(sprintf('Image Plane Cam | Galvo tilt %i/%i\nsaturation = %.0f%%', g, G, saturation_img))
            colorbar
        end
        
        fprintf(repmat('\b', [1 numchars]))
        numchars = fprintf('Galvo tilt %i/%i\n', g, G);
    end
    
    if dosave
        % Save that stuff! Save for each segment position separately to prevent massive files
        savepath = fullfile(p.savedir, sprintf('%s_%03i.mat', p.savename, s));
        disp('Saving...')
        save(savepath, '-v7.3', 'frames_ft', 'frames_img', 'darkframe_img', 'darkframe_ft', 'frame_ft_sum', 'frame_img_sum', ...
            's', 'p', 'sopt', 'copt_ft', 'copt_img')
    end
    
    eta(sg, SG, starttime, 'console', ...
        sprintf('Measuring pencil beams...\nRaw data size will be: %s\nSegments done: %i/%i',rawdatasize,s,S), 0);
end


if dosave
    disp('Moving save files to network data folder...')
    p.networkdatadir = fullfile([dirs.expdata p.subdir], p.savename);
    movefile(p.savedir, p.networkdatadir)
    fprintf('\nDone moving files to:\n%s\n', p.networkdatadir)
end
