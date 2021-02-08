%% Test to fit 4D polynomial, to model camera coords to SLM/Galvo coords

forcereset = 0;

if forcereset || ~exist('dirs', 'var')
    close all; clc; clear

    % Define location paths
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
    dirconfig_raylearn
end
    calibration_filename = '03-Nov-2020-empty/raylearn_pencil_beam_738098.472397_empty.mat';
    load(fullfile(dirs.localdata, 'TPM/pencil-beam-positions' , calibration_filename));
%     load /home/daniel/d/ScientificData/raylearn-data/TPM/pencil-beam-positions/03-Nov-2020-empty/raylearn_pencil_beam_738098.472397_empty.mat
% load F:\ScientificData\raylearn-data\TPM\pencil-beam-positions\03-Nov-2020-empty\raylearn_pencil_beam_738098.472397_empty.mat

%% Load SLM and Galvo coordinates of calibration
Yslm_input   = repmat(p.rects(:,1), [size(cam_ft_col, 2) 1]);
Xslm_input   = repmat(p.rects(:,2), [size(cam_ft_row, 2) 1]);
Ygalvo_input = repelem(p.galvoYs, size(cam_img_col, 1));
Xgalvo_input = repelem(p.galvoXs, size(cam_img_row, 1));

x  = cam_img_row(:);
y  = cam_img_col(:);
kx = cam_ft_col(:);
ky = cam_ft_row(:);

%%%% Turn it into mat file object
measurement_filename = '03-Nov-2020-400um/raylearn_pencil_beam_738098.478067_400um.mat';
load(fullfile(dirs.localdata, 'TPM/pencil-beam-positions', measurement_filename));

x_query  = cam_img_row(:);
y_query  = cam_img_col(:);
kx_query = cam_ft_col(:);
ky_query = cam_ft_row(:);

%% kx&ky-plane around average x&y

[Xslm_output, Yslm_output, Xgalvo_output, Ygalvo_output] = ...
    griddatan_4Doutput([x, y, kx, ky], Xslm_input, Yslm_input, Xgalvo_input, Ygalvo_input,...
    [x_query(:) y_query(:) kx_query(:) ky_query(:)]);

%%
savepath = fullfile(dirs.localdata, '/TPM/pencil-beam-interpolated', measurement_filename);
warning('off', 'MATLAB:MKDIR:DirectoryExists');
try mkdir(fileparts(savepath)); catch 'MATLAB:MKDIR:DirectoryExists'; end   % Create savedir if needed
save(savepath, '-v7.3', 'p', 'Yslm_input', 'Xslm_input', 'Ygalvo_input', 'Xgalvo_input',...
    'Xslm_output', 'Yslm_output', 'Xgalvo_output', 'Ygalvo_output')
