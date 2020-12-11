%% Test to fit 4D polynomial, to model camera coords to SLM/Galvo coords

doreset = 0;

if doreset
    close all; clc; clear

    % Define location paths
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Parent dir
    dirconfig_raylearn

%     load(fullfile(dirs.localdata, 'pencil-beam-positions/03-Nov-2020-empty/raylearn_pencil_beam_738098.472397_empty.mat'));
    load F:\ScientificData\raylearn-data\TPM\pencil-beam-positions\03-Nov-2020-empty\raylearn_pencil_beam_738098.472397_empty.mat
end
    
%%
camsize_pix = 1024;
Npoints = numel(cam_ft_col);
randorder = randperm(Npoints);
Nfitset = floor(Npoints*4/5);
fitset  = randorder(1:Nfitset);
testset = randorder((Nfitset+1):end);

x  = cam_img_row(:) * 4/camsize_pix - 2;
y  = cam_img_col(:) * 4/camsize_pix - 2;
kx = cam_ft_col(:)  * 4/camsize_pix - 2;
ky = cam_ft_row(:)  * 4/camsize_pix - 2;

%% Create basis polynomial functions
% Create coordinate arrays for x,y,kx,ky

% Create 1D arrays containing 1, x, x^2, ..., 1, y, y^2, ..., 1, kx, kx^2, ...
npowers = 6;                                    % Polynomial powers (including 0)
orderlabels = cell(npowers.^4);
powers = 0:(npowers-1);                         % Array containing all powers

xpowers  =  x.^powers;
ypowers  =  y.^powers;
kxpowers = kx.^powers;
kypowers = ky.^powers;

xykxky_cam_basis = zeros(Npoints, npowers.^4);  % Initialize basis


%% Prepare interpolation data
% Ninterppoints = 100;
% xint  = linspace(min(x),  max(x),  Ninterppoints);
% yint  = linspace(min(y),  max(y),  Ninterppoints);
% kxint = linspace(min(kx), max(kx), Ninterppoints);
% kyint = linspace(min(ky), max(ky), Ninterppoints);
% 
% xintpowers  =  xint.^powers;
% yintpowers  =  yint.^powers;
% kxpowers = kx.^powers;
% kypowers = ky.^powers;
% 
% xykxky_interp_basis


%% Loop over all powers in x,y,kx,ky
m = 1;
px = 0;
for xpow = xpowers
    py = 0;
    for ypow = ypowers
        pkx = 0;
        for kxpow = kxpowers
            pky = 0;
            for kypow = kypowers
                % Add 4D polynomial to set of basis functions
                xykxky_cam_basis(:, m) = xpow .* ypow .* kxpow .* kypow;
%                 xykxky_interp_basis(:, m) = 
                
                orderlabels{m} = sprintf("x^%iy^%ik_x^%ik_y^%i", px, py, pkx, pky);
                m = m+1;
                pky = pky+1;
            end
            pkx = pkx+1;
        end
        py = py+1;
    end
    px = px+1;
end

xykxky_cam_basis_fit = xykxky_cam_basis(fitset,  :);
xykxky_cam_basis_test = xykxky_cam_basis(testset, :);

%% Create test data from coefficients
Yslm_gt = repmat(p.rects(:,1), [size(cam_ft_col, 2) 1]);
Yslm_cf   = xykxky_cam_basis_fit \ Yslm_gt(fitset);                 % Compute coefficients
Yslm_fit  = xykxky_cam_basis_fit  * Yslm_cf;                % Compute fit
Yslm_test = xykxky_cam_basis_test * Yslm_cf;

Xslm_gt = repmat(p.rects(:,2), [size(cam_ft_row, 2) 1]);
Xslm_cf   = xykxky_cam_basis_fit \ Xslm_gt(fitset);                 % Compute coefficients
Xslm_fit  = xykxky_cam_basis_fit  * Xslm_cf;                % Compute fit
Xslm_test = xykxky_cam_basis_test * Xslm_cf;

Ygalvo_gt = repelem(p.galvoYs, size(cam_img_col, 1));
Ygalvo_cf   = xykxky_cam_basis_fit \ Ygalvo_gt(fitset);                 % Compute coefficients
Ygalvo_fit  = xykxky_cam_basis_fit * Ygalvo_cf;                % Compute fit
Ygalvo_test = xykxky_cam_basis_test * Ygalvo_cf;

Xgalvo_gt = repelem(p.galvoXs, size(cam_img_row, 1));
Xgalvo_cf   = xykxky_cam_basis_fit \ Xgalvo_gt(fitset);                 % Compute coefficients
Xgalvo_fit  = xykxky_cam_basis_fit  * Xgalvo_cf;                % Compute fit
Xgalvo_test = xykxky_cam_basis_test * Xgalvo_cf;                % Compute test

%% Test
npowers
Yslm_NRMSE   = sqrt(mean((Yslm_test   - Yslm_gt(testset)).^2))   / mean(abs(Yslm_gt(testset)))
Xslm_NRMSE   = sqrt(mean((Xslm_test   - Xslm_gt(testset)).^2))   / mean(abs(Xslm_gt(testset)))
Ygalvo_NRMSE = sqrt(mean((Ygalvo_test - Ygalvo_gt(testset)).^2)) / mean(abs(Ygalvo_gt(testset)))
Xgalvo_NRMSE = sqrt(mean((Xgalvo_test - Xgalvo_gt(testset)).^2)) / mean(abs(Xgalvo_gt(testset)))

%% Plot
% close all

% Scatter plot matrix
figure
set(0, 'DefaultAxesFontSize', 14)
[~, ax] = plotmatrix([x y kx ky], [Xgalvo_gt Ygalvo_gt Xslm_gt Yslm_gt]);
title('Correlations')

ax(1,1).YLabel.String='Xgalvo';
ax(2,1).YLabel.String='Ygalvo';
ax(3,1).YLabel.String='Xslm';
ax(4,1).YLabel.String='Yslm';

ax(4,1).XLabel.String='x';
ax(4,2).XLabel.String='y';
ax(4,3).XLabel.String='k_x';
ax(4,4).XLabel.String='k_y';

% Scatter plot x & y
figure
plot(cam_img_row', cam_img_col', '.')
title('Image plane points')
xlabel('row coordinate (pix)')
ylabel('column coordinate (pix)')

% Plot slices of fit data vs ground truth
figure
subplot(1,2,1)
plot(Xgalvo_fit, Ygalvo_fit, '+b')
hold on
plot(Xgalvo_gt, Ygalvo_gt, 'or')
axis image
title('Fit Galvo')
xlabel('Xgalvo')
ylabel('Ygalvo')
legend('Fit', 'Ground Truth')
hold off

subplot(1,2,2)
plot(Xslm_fit, Yslm_fit, '+b')
hold on
plot(Xslm_gt, Yslm_gt, 'or')
axis image
title('Fit SLM')
xlabel('Xslm')
ylabel('Yslm')
legend('Fit', 'Ground Truth')
hold off

% Plot bar graphs of coefficients
figure
hbar = bar(Yslm_cf);    % Create bar plot
title('Y_{SLM} coefficients')
xlabel('index')
ylabel('coefficient')
% Get the data for all the bars that were plotted
xbar = get(hbar,'XData');
ybar = get(hbar,'YData');
for i = 1:length(xbar) % Loop over each bar
    if abs(Yslm_cf(i)) > 0.4*mean(abs(Yslm_cf))
        htext = text(xbar(i),ybar(i),orderlabels{i});          % Add text label
        set(htext,'VerticalAlignment','middle',...  % Adjust properties
                  'HorizontalAlignment','center')
    end
end


% figure(3)
% xfit  = linspace(min(x), max(x), 100);
% yfit  = mean(y);
% kxfit = mean(kx);
% kyfit = mean(ky);
% 
