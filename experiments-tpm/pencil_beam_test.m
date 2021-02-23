doreset = 1;

if doreset
    close all; clear; clc
    setup_raylearn_exptpm
end

bg_patch_id = 2;                    % Background grating Patch ID
segment_patch_id = 3;               % Pencil Beam segment SLM patch ID

diameter = 0.30;                    % diameter of circular SLM geometry
slm_offset_x = 0.00;                % horizontal offset of rectangle SLM geometry
slm_offset_y = 0.00;                % vertical offset of rectangle SLM geometry
N_diameter = 8;                     % number of segments on SLM diameter

% Set background grating
[NxSLM, NySLM] = size(slm.getPixels);
slm.setRect(bg_patch_id, [0 0 1 1]);
slm.setData(bg_patch_id, bg_grating('blaze', -12, 0, 255, NySLM)');
slm.update


%% Move galvos
% X = -0.14;  % left
% X = -0.08;  % right
X = -0.11*1.0; % center
% Y = 0.08; % bottom
% Y = -0.02; % top
Y = 0.03; % center
outputSingleScan(daqs, [X,Y]);