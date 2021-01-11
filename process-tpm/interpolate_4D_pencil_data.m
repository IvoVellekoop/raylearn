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


%% kx&ky-plane around average x&y
Nquery = 40;
[x_query, y_query, kx_query, ky_query] = ...
    ndgrid(mean(x), mean(y), linspace(min(kx), max(kx), Nquery), linspace(min(ky), max(ky), Nquery));

tic
[Xslm_interp, Yslm_interp, Xgalvo_interp, Ygalvo_interp] = ...
    griddatan_4Doutput([x, y, kx, ky], Xslm_gt, Yslm_gt, Xgalvo_gt, Ygalvo_gt,...
    [x_query(:) y_query(:) kx_query(:) ky_query(:)]);
toc

%% Plot kx&ky-surface
XData = reshape(kx_query, [Nquery, Nquery]);
YData = reshape(ky_query, [Nquery, Nquery]);

figure(1)
subplot(2,2,1)
surf(XData, YData, reshape(Xslm_interp, [Nquery, Nquery]))
xlabel('kx (pixels)'); ylabel('ky (pixels)')
title('Xslm')

subplot(2,2,2)
surf(XData, YData, reshape(Yslm_interp, [Nquery, Nquery]))
xlabel('kx (pixels)'); ylabel('ky (pixels)')
title('Yslm')

subplot(2,2,3)
surf(XData, YData, reshape(Xgalvo_interp, [Nquery, Nquery]))
xlabel('kx (pixels)'); ylabel('ky (pixels)')
title('Xgalvo')

subplot(2,2,4)
surf(XData, YData, reshape(Ygalvo_interp, [Nquery, Nquery]))
xlabel('kx (pixels)'); ylabel('ky (pixels)')
title('Ygalvo')



%% x&y-plane around average kx&ky
Nquery = 40;
[x_query, y_query, kx_query, ky_query] = ...
    ndgrid(linspace(min(x), max(x), Nquery), linspace(min(y), max(y), Nquery), mean(kx), mean(ky));

tic
[Xslm_interp, Yslm_interp, Xgalvo_interp, Ygalvo_interp] = ...
    griddatan_4Doutput([x, y, kx, ky], Xslm_gt, Yslm_gt, Xgalvo_gt, Ygalvo_gt,...
    [x_query(:) y_query(:) kx_query(:) ky_query(:)]);
toc

%% Plot x&y-surface
XData = reshape(x_query, [Nquery, Nquery]);
YData = reshape(y_query, [Nquery, Nquery]);

figure(2)
subplot(2,2,1)
surf(XData, YData, reshape(Xslm_interp, [Nquery, Nquery]))
xlabel('x (pixels)'); ylabel('y (pixels)')
title('Xslm')

subplot(2,2,2)
surf(XData, YData, reshape(Yslm_interp, [Nquery, Nquery]))
xlabel('x (pixels)'); ylabel('y (pixels)')
title('Yslm')

subplot(2,2,3)
surf(XData, YData, reshape(Xgalvo_interp, [Nquery, Nquery]))
xlabel('x (pixels)'); ylabel('y (pixels)')
title('Xgalvo')

subplot(2,2,4)
surf(XData, YData, reshape(Ygalvo_interp, [Nquery, Nquery]))
xlabel('x (pixels)'); ylabel('y (pixels)')
title('Ygalvo')



%% y&kx-plane around average x&ky
Nquery = 40;
[x_query, y_query, kx_query, ky_query] = ...
    ndgrid(mean(x), linspace(min(y), max(y), Nquery), linspace(min(kx), max(kx), Nquery), mean(ky));

tic
[Xslm_interp, Yslm_interp, Xgalvo_interp, Ygalvo_interp] = ...
    griddatan_4Doutput([x, y, kx, ky], Xslm_gt, Yslm_gt, Xgalvo_gt, Ygalvo_gt,...
    [x_query(:) y_query(:) kx_query(:) ky_query(:)]);
toc

%% Plot y&kx-surface
XData = reshape(y_query, [Nquery, Nquery]);
YData = reshape(kx_query, [Nquery, Nquery]);

figure(3)
subplot(2,2,1)
surf(XData, YData, reshape(Xslm_interp, [Nquery, Nquery]))
xlabel('y (pixels)'); ylabel('kx (pixels)')
title('Xslm')

subplot(2,2,2)
surf(XData, YData, reshape(Yslm_interp, [Nquery, Nquery]))
xlabel('y (pixels)'); ylabel('kx (pixels)')
title('Yslm')

subplot(2,2,3)
surf(XData, YData, reshape(Xgalvo_interp, [Nquery, Nquery]))
xlabel('y (pixels)'); ylabel('kx (pixels)')
title('Xgalvo')

subplot(2,2,4)
surf(XData, YData, reshape(Ygalvo_interp, [Nquery, Nquery]))
xlabel('y (pixels)'); ylabel('kx (pixels)')
title('Ygalvo')


