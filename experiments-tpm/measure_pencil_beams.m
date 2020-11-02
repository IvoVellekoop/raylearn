%% Measure pencil beams
% Use The Galvo Mirrors and the SLM to create pencil beams
% with various angles and record the outgoing pencil beams.

%% Reset variables and hardware connections
doreset = 0;

if doreset
    close all; clear; clc
    setup_raylearn_exptpm
end

%% Settings
p.samplename = '2x170um';
doshowcams = 0;                     % Toggle show what the cameras see
dosave = 1;                         % Toggle savings
dochecksamplename = 0;              % Toggle console sample name check

% SLM Settings
p.ppp = 12;                         % Pixels per period for the grating. Should match Galvo setting!
p.segmentsize_pix = 6 * p.ppp;      % Segment width in pixels
p.N_diameter = 9;                   % Number of segments across SLM diameter
p.beamdiameter = 0.60;              % Diameter of circular SLM segment set (relative coords)
p.slm_offset_x = 0.00;              % Horizontal offset of rectangle SLM geometry (relative coords)
p.slm_offset_y = 0.00;              % Vertical offset of rectangle SLM geometry (relative coords)
p.segment_patch_id = 2;             % Pencil Beam segment SLM patch ID

% Galvo Mirror settings
p.GalvoXcenter = -0.11;             % Galvo center x
p.GalvoYcenter = 0.04;              % Galvo center y
p.GalvoRadius = 0.015*0;              % Galvo scan radius: from center to outer
p.GalvoNX = 1;                      % Number of Galvo steps, x
p.GalvoNY = 1;                      % Number of Galvo steps, y

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
p.blaze = single(bg_grating('blaze', -p.ppp, 0, 255, p.segmentsize_pix)' * ones(1, p.segmentsize_pix));
p.blaze(mask_outside_circle) = 0;                           % Set pixels outside circle to 0

% Create set of SLM segment rectangles within circle
[p.rects, N] = BlockedCircleSegments(p.N_diameter, p.beamdiameter, p.slm_offset_x, p.slm_offset_y, p.segmentwidth, p.segmentheight);

% Create set of Galvo tilts
p.galvoXs1d = linspace(p.GalvoXcenter-p.GalvoRadius, p.GalvoXcenter+p.GalvoRadius, p.GalvoNX);
p.galvoYs1d = linspace(p.GalvoYcenter-p.GalvoRadius, p.GalvoYcenter+p.GalvoRadius, p.GalvoNY);
[galvoXsq, galvoYsq] = meshgrid(p.galvoXs1d, p.galvoYs1d);
galvo_scan_mask = ((galvoXsq-p.GalvoXcenter).^2 + (galvoYsq-p.GalvoYcenter).^2 <= p.GalvoRadius.^2);
p.galvoXs = single(galvoXsq(galvo_scan_mask));
p.galvoYs = single(galvoYsq(galvo_scan_mask));
G = numel(p.galvoXs);


%% Initialize arrays
frames_ft  = zeros(copt_img.Width, copt_img.Height, N, G, 'single');
frames_img = zeros(copt_img.Width, copt_img.Height, N, G, 'single');


% %% Capture dark frames
% outputSingleScan(daqs, [p.GalvoXcenter, p.GalvoYcenter]);   % Set Galvo
% slm.setData(p.segment_patch_id, 0);                         % Set SLM segment pixels to 0
% slm.update;
% 
% cam_ft.trigger;
% darkframe_ft = cam_ft.getData;                              % Fourier Plane camera dark frame
% cam_img.trigger;
% darkframe_img = cam_img.getData;                            % Image Plane camera dark frame


%% Main measurement loop
slm.setData(p.segment_patch_id, p.blaze);

if doshowcams                       % Spawn wide figure
    figure; plot(0,0);
    fig_resize(400, 2.5);
end

starttime = now;
NG = N*G;
ng = 0;
for n = 1:N                        % Loop over SLM segments
    for g = 1:G                     % Loop over Galvo tilts
        ng = ng+1;                  % Counter for ETA
        outputSingleScan(daqs, [p.galvoXs(g), p.galvoYs(g)]);   % Set Galvo Mirror
        pause(0.005)
    
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
            title(sprintf('Fourier Plane Cam | Segment %i/%i\nsaturation = %.0f%%', n, N, saturation_ft))
            colorbar

            subplot(1,2,2)
            imagesc(frame_img)
            saturation_img = 100 * max(frame_img,[],'all') / 65520; % 12bit -> 2^16 - 2^4 = 65520
            title(sprintf('Image Plane Cam | Galvo tilt %i/%i\nsaturation = %.0f%%', g, G, saturation_img))
            colorbar
        end
        
        eta(ng, NG, starttime, 'console', sprintf('Measuring pencil beams...\nSegment: %i/%i, Tilt: %i/%i',n,N,g,G), 0);
    end
end

if dosave
    % Save that stuff! Save for each segment position separately to prevent massive files
    p = preparesave(p, dirs, sprintf('raylearn_pencil_beam'));
    save(p.savepath, '-v7.3', 'frames_ft', 'frames_img', 'darkframe_ft', 'darkframe_img', 'n', 'p', 'sopt', 'copt_ft', 'copt_img')
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
    p.savepath = fullfile(savedir, sprintf('%s_%f_%s.mat', prefix, now, p.samplename));
    fprintf('\nSaving to %s\n', p.savepath)
end
