%% Measure pencil beams
% Use The Galvo Mirrors and the SLM to create pencil beams
% with various angles and record the outgoing pencil beams.

%% Reset variables and hardware connections
doreset = 1;

if doreset
    close all; clear; clc
    setup_raylearn_exptpm
end

%% Settings
p.samplename = 'newsample';
doshowcams = 1;                     % Show what the cameras see
dosave = 1;                         % Toggle savings

% SLM Settings
p.ppp = 12;                         % Pixels per period for the grating. Should match Galvo setting!
p.segmentsize_pix = 4 * p.ppp;      % Segment width in pixels
p.N_diameter = 8;                   % Number of segments across SLM diameter
p.beamdiameter = 0.60;              % Diameter of circular SLM segment set (relative coords)
p.slm_offset_x = 0.00;              % Horizontal offset of rectangle SLM geometry (relative coords)
p.slm_offset_y = 0.00;              % Vertical offset of rectangle SLM geometry (relative coords)
p.segment_patch_id = 2;             % Pencil Beam segment SLM patch ID

% Galvo Mirror settings
p.GalvoXcenter = -0.091;            % Galvo center x
p.GalvoYcenter = 0;                 % Galvo center y
p.GalvoXmax = 0.015;                % Galvo center to outer, x
p.GalvoYmax = 0.015;                % Galvo center to outer, y
p.GalvoNX = 5;                      % Number of Galvo steps, x
p.GalvoNY = 5;                      % Number of Galvo steps, y

% Ask if samplename is correct
fprintf('\nSample name set to <strong>%s</strong>. Is this correct?\nPress Enter to continue...\n', p.samplename)
pause


%% Compute constants
% Compute segment size in relative coords

% [p.NxSLM, p.NySLM] = size(slm.getPixels);                 % Retrieve SLM size in pixels
p.NySLM = 1080;
p.segmentwidth = p.segmentsize_pix / p.NySLM;               % Segment width in relative coords
p.segmentheight = p.segmentwidth;                           % Segment height in relative coords

% Create blaze grated segment, masked with circle
xblaze = linspace(-1,1,p.segmentsize_pix);                  % Create x coords for segment pixels
yblaze = xblaze';                                           % Create x coords for segment pixels
blazeradius = (p.segmentsize_pix+1) / p.segmentsize_pix;    % Radius from center to borderpixel edge
mask_outside_circle = (xblaze.^2+yblaze.^2) > blazeradius;  % Mask for pixels outside circle
p.blaze = single(bg_grating('blaze', -p.ppp, 0, 255, p.segmentsize_pix)' * ones(1, p.segmentsize_pix));
p.blaze(mask_outside_circle) = 0;                           % Set pixels outside circle to 0

% Create set of SLM segment rectangles within circle
[p.rects, N] = BlockedCircleSegments(p.N_diameter, p.beamdiameter, p.slm_offset_x, p.slm_offset_y, p.segmentwidth, p.segmentheight);

% Create set of Galvo tilts
p.galvoXs1d = linspace(p.GalvoXcenter-p.GalvoXmax, p.GalvoXcenter+p.GalvoXmax, p.GalvoNX);
p.galvoYs1d = linspace(p.GalvoYcenter-p.GalvoYmax, p.GalvoYcenter+p.GalvoYmax, p.GalvoNY);
[galvoXsq, galvoYsq] = single(meshgrid(p.galvoXs1d, p.galvoYs1d));
p.galvoXs = galvoXsq(:);
p.galvoYs = galvoYsq(:);
G = p.GalvoNX * p.GalvoNY;

%% Initialize arrays
frames_ft  = zeros(copt_img.Width, copt_img.Height, G, 'single');
frames_img = zeros(copt_img.Width, copt_img.Height, G, 'single');


%% Capture dark frames
outputSingleScan(daqs, [p.GalvoXcenter, p.GalvoYcenter]);   % Set Galvo
slm.setData(p.segment_patch_id, 0);                         % Set SLM segment pixels to 0
slm.update;

cam_ft.trigger;
darkframe_ft = cam_ft.getData;                              % Fourier Plane camera dark frame
cam_img.trigger;
darkframe_img = cam_img.getData;                            % Image Plane camera dark frame

if dosave
    savepath = preparesave(p, sprintf('raylearn_dark_frames'));
    save(savepath, '-v7.3',...
        'darkframe_ft', 'darkframe_img', 'p', 'sopt', 'copt_ft', 'copt_img')
end


%% Main measurement loop
slm.setData(p.segment_patch_id, p.blaze);

for n = 1:N                         % Loop over SLM segments
    for g = 1:G                     % Loop over Galvo tilts
        outputSingleScan(daqs, [p.galvoXs, p.galvoYs]);   % Set Galvo Mirror
        pause(0.01)
    
        % Incoming angle of pencil beam: set SLM segment
        slm.setRect(p.segment_patch_id, p.rects(n,:));
        slm.update

        % Capture camera frames
        cam_ft.trigger;
        frame_ft = single(cam_ft.getData);
        frames_ft(:,:,n,g) = frame_ft;

        cam_img.trigger;
        frame_img = single(cam_img.getData);
        frames_img(:,:,n,g) = frame_img;

        % Show what's on the cameras and check for overexposure
        if doshowcams
            subplot(1,2,1)
            imagesc(frame_ft)
            saturation_ft = 100 * max(frame_ft,[],'all') / 65520;   % 12bit -> 2^16 - 2^4 = 65520
            title(sprintf('Fourier Plane Cam\nsaturation = %.0f%%', saturation_ft))
            colorbar

            subplot(1,2,2)
            imagesc(frame_img)
            saturation_img = 100 * max(frame_img,[],'all') / 65520; % 12bit -> 2^16 - 2^4 = 65520
            title(sprintf('Image Plane Cam\nsaturation = %.0f%%', saturation_img))
            colorbar
        end
    
    end
    
    if dosave
        % Save that stuff! Save for each segment position separately to prevent massive files
        p = preparesave(p, dirs, sprintf('raylearn_pencil_beam_%i', n));
        save(p.savepath, '-v7.3', 'frames_ft', 'frames_img', 'n', 'p', 'sopt', 'copt_ft', 'copt_img')
    end
end


%% Subfunctions

function p = preparesave(p, dirs, prefix)
    % Script path, revision and date&time
    p.script_name = mfilename('fullpath');
    [~, p.script_version] = system(['git --git-dir="' dirs.repo '\.git" rev-parse HEAD']);
    [~, p.git_diff] = system(['git --git-dir="' dirs.repo '\.git" diff']);
    p.save_time = now;

    % Save
    savedir = [dirs.localdata '\raylearn-data\TPM\' date '-' p.samplename];
    try mkdir(savedir); catch 'MATLAB:MKDIR:DirectoryExists'; end
    p.savepath = fullfile(savedir, '%s_%f_%s.mat', prefix, now, p.samplename);
    fprintf('\nSaving to %s\n', p.savepath)
end
