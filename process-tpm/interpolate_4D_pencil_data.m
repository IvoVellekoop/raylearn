%% Test to fit 4D polynomial, to model camera coords to SLM/Galvo coords

forcereset = 0;

if forcereset || ~exist('cam_img_row', 'var')
    close all; clc; clear

    % Define location paths
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
    dirconfig_raylearn

%     load(fullfile(dirs.localdata, 'pencil-beam-positions/03-Nov-2020-empty/raylearn_pencil_beam_738098.472397_empty.mat'));
%     load /home/daniel/d/ScientificData/raylearn-data/TPM/pencil-beam-positions/03-Nov-2020-empty/raylearn_pencil_beam_738098.472397_empty.mat
    load F:\ScientificData\raylearn-data\TPM\pencil-beam-positions\03-Nov-2020-empty\raylearn_pencil_beam_738098.472397_empty.mat
end

%% Load SLM and Galvo coordinates
Yslm_gt   = repmat(p.rects(:,1), [size(cam_ft_col, 2) 1]);
Xslm_gt   = repmat(p.rects(:,2), [size(cam_ft_row, 2) 1]);
Ygalvo_gt = repelem(p.galvoYs, size(cam_img_col, 1));
Xgalvo_gt = repelem(p.galvoXs, size(cam_img_row, 1));


%%
test_percentage = 10;
camsize_pix = 1024;
Npoints = numel(cam_ft_col);
randorder = randperm(Npoints);
Nfitset = floor(Npoints * (100-test_percentage)/100);
fitset  = randorder(1:Nfitset);
testset = randorder((Nfitset+1):end);

x  = cam_img_row(:);
y  = cam_img_col(:);
kx = cam_ft_col(:);
ky = cam_ft_row(:);



Nquery = 40;
[x_query, y_query, kx_query, ky_query] = ...
    ndgrid(mean(x), mean(y), linspace(min(kx), max(kx), Nquery), linspace(min(ky), max(ky), Nquery));


tic
[Xslm_interp, Yslm_interp, Xgalvo_interp, Ygalvo_interp] = ...
    griddatan_4Doutput([x, y, kx, ky], Xslm_gt, Yslm_gt, Xgalvo_gt, Ygalvo_gt,...
    [x_query(:) y_query(:) kx_query(:) ky_query(:)]);
toc

tic
Xslm_interp_orig = griddatan([x, y, kx, ky], Xslm_gt, [x_query(:) y_query(:) kx_query(:) ky_query(:)]);
Yslm_interp_orig = griddatan([x, y, kx, ky], Yslm_gt, [x_query(:) y_query(:) kx_query(:) ky_query(:)]);
Xgalvo_interp_orig = griddatan([x, y, kx, ky], Xgalvo_gt, [x_query(:) y_query(:) kx_query(:) ky_query(:)]);
Ygalvo_interp_orig = griddatan([x, y, kx, ky], Ygalvo_gt, [x_query(:) y_query(:) kx_query(:) ky_query(:)]);
toc


isequaln(Xslm_interp_orig, Xslm_interp)
isequaln(Yslm_interp_orig, Yslm_interp)
isequaln(Xgalvo_interp_orig, Xgalvo_interp)
isequaln(Ygalvo_interp_orig, Ygalvo_interp)











