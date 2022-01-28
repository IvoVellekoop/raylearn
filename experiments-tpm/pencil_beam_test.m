doreset = 0;

if doreset || ~exist('slm', 'var') || ~exist('daqs', 'var')  || ~exist('cam_ft', 'var')
    close all; clear; clc
    setup_raylearn_exptpm
end

p.bg_patch_id = 2;                    % Background grating Patch ID
p.segment_patch_id = 3;               % Pencil Beam segment SLM patch ID
p.ppp = 2;
p.beamdiameter = 1.00;                    % diameter of circular SLM geometry
p.slm_offset_x = 0.00;                % horizontal offset of rectangle SLM geometry
p.slm_offset_y = 0.00;                % vertical offset of rectangle SLM geometry

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


% Set background grating
[NxSLM, NySLM] = size(slm.getPixels);
slm.setRect(p.bg_patch_id, [0 0 1 1]);
slm.setData(p.bg_patch_id, bg_grating('blaze', p.ppp, 0, 255, NySLM)');
slm.update


%% Move galvos
% X = -0.14;  % left
% X = -0.08;  % right
X = 0.0; % center
% Y = 0.08; % bottom
% Y = -0.02; % top
Y = 0.0; % center
outputSingleScan(daqs, [X,Y]);
