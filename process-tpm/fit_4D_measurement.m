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

%% Load SLM and Galvo coordinates
Yslm_gt   = repmat(p.rects(:,1), [size(cam_ft_col, 2) 1]);
Xslm_gt   = repmat(p.rects(:,2), [size(cam_ft_row, 2) 1]);
Ygalvo_gt = repelem(p.galvoYs, size(cam_img_col, 1));
Xgalvo_gt = repelem(p.galvoXs, size(cam_img_row, 1));


%%
test_percentage = 20;
camsize_pix = 1024;
Npoints = numel(cam_ft_col);
randorder = randperm(Npoints);
Nfitset = floor(Npoints * (100-test_percentage)/100);
fitset  = randorder(1:Nfitset);
testset = randorder((Nfitset+1):end);

x  = cam_img_row(:) * 4/camsize_pix - 2;
y  = cam_img_col(:) * 4/camsize_pix - 2;
kx = cam_ft_col(:)  * 4/camsize_pix - 2;
ky = cam_ft_row(:)  * 4/camsize_pix - 2;

%% Sort into 4D grid for easy access
% Turn SLM and Galvo coords from linear array into 4D array; fill corners with NaNs.

% Generate grid indices per dimension
index_Xslm_grid   = map2integers(Xslm_gt);
index_Yslm_grid   = map2integers(Yslm_gt);
index_Xgalvo_grid = map2integers(Xgalvo_gt);
index_Ygalvo_grid = map2integers(Ygalvo_gt);

% Size of 4D grid array
sz = [max(index_Xslm_grid) max(index_Yslm_grid) max(index_Xgalvo_grid) max(index_Ygalvo_grid)];

% Convert 4D grid indices to linear indices
index_lin = sub2ind(sz, index_Xslm_grid, index_Yslm_grid, index_Xgalvo_grid, index_Ygalvo_grid);

% Create 4D grid version of SLM and Galvo coordinate arrays
Xslm_gt_grid   = lindex2grid(Xslm_gt,   index_lin, sz);
Yslm_gt_grid   = lindex2grid(Yslm_gt,   index_lin, sz);
Xgalvo_gt_grid = lindex2grid(Xgalvo_gt, index_lin, sz);
Ygalvo_gt_grid = lindex2grid(Ygalvo_gt, index_lin, sz);

x_grid  = lindex2grid(x,  index_lin, sz);
y_grid  = lindex2grid(y,  index_lin, sz);
kx_grid = lindex2grid(kx, index_lin, sz);
ky_grid = lindex2grid(ky, index_lin, sz);


%%%%
index_xslm = 6;
index_yslm = 5;
index_xgalvo = 3;
index_ygalvo = 3;
plot(squeeze(Xgalvo_gt_grid(index_xslm,index_yslm,:,index_ygalvo))', squeeze(kx_grid(index_xslm,index_yslm,:,index_ygalvo))', '.-')
%%%%

%% 


%% Create basis polynomial functions
% Create coordinate arrays for x,y,kx,ky

% for npowers = 1:7
for npowers = 4

    % Create 1D arrays containing 1, x, x^2, ..., 1, y, y^2, ..., 1, kx, kx^2, ...
    % npowers = 6;                                    % Polynomial powers (including 0)
    orderlabels = cell(npowers.^4);
    powers = 0:(npowers-1);                         % Array containing all powers

    xpowers  =  x.^powers;
    ypowers  =  y.^powers;
    kxpowers = kx.^powers;
    kypowers = ky.^powers;

    xykxky_cam_basis = zeros(Npoints, npowers.^4);  % Initialize basis


    %% Prepare interpolation data
%     index_xslm = 6;
%     index_yslm = 5;
%     index_xgalvo = 3;
%     index_ygalvo = 3;
%     x_grid(index_xslm, index_yslm, : ,index_ygalvo)
    
%     Ninterppoints = 100;
%     xint  = linspace(min(x),  max(x),  Ninterppoints);
%     yint  = linspace(min(y),  max(y),  Ninterppoints);
%     kxint = linspace(min(kx), max(kx), Ninterppoints);
%     kyint = linspace(min(ky), max(ky), Ninterppoints);
%     
%     xintpowers  =  xint.^powers;
%     yintpowers  =  yint.^powers;
%     kxpowers = kx.^powers;
%     kypowers = ky.^powers;
%     
%     xykxky_interp_basis = zeros(Npoints, npowers.^4);


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
    Yslm_cf   = xykxky_cam_basis_fit \ Yslm_gt(fitset);                 % Compute coefficients
    Yslm_fit  = xykxky_cam_basis  * Yslm_cf;                % Compute fit
    Yslm_test = xykxky_cam_basis_test * Yslm_cf;

    Xslm_cf   = xykxky_cam_basis_fit \ Xslm_gt(fitset);                 % Compute coefficients
    Xslm_fit  = xykxky_cam_basis  * Xslm_cf;                % Compute fit
    Xslm_test = xykxky_cam_basis_test * Xslm_cf;
    
    Ygalvo_cf   = xykxky_cam_basis_fit \ Ygalvo_gt(fitset);                 % Compute coefficients
    Ygalvo_fit  = xykxky_cam_basis * Ygalvo_cf;                % Compute fit
    Ygalvo_test = xykxky_cam_basis_test * Ygalvo_cf;

    Xgalvo_cf   = xykxky_cam_basis_fit \ Xgalvo_gt(fitset);                 % Compute coefficients
    Xgalvo_fit  = xykxky_cam_basis  * Xgalvo_cf;                % Compute fit
    Xgalvo_test = xykxky_cam_basis_test * Xgalvo_cf;                % Compute test

    
    %% Test overfitting with test set
    Yslm_NRMSE(npowers)   = sqrt(mean((Yslm_test   - Yslm_gt(testset)).^2))   / std(Yslm_gt(testset));
    Xslm_NRMSE(npowers)   = sqrt(mean((Xslm_test   - Xslm_gt(testset)).^2))   / std(Xslm_gt(testset));
    Ygalvo_NRMSE(npowers) = sqrt(mean((Ygalvo_test - Ygalvo_gt(testset)).^2)) / std(Ygalvo_gt(testset));
    Xgalvo_NRMSE(npowers) = sqrt(mean((Xgalvo_test - Xgalvo_gt(testset)).^2)) / std(Xgalvo_gt(testset));
    
    
    %% Select 1D lines for analyzing overfitting %%%%
    Xline_condition = (Yslm_gt==0) & (Xslm_gt==0) & (Ygalvo_gt<0.0401) & (Ygalvo_gt>0.0399);
    Xgalvo_1D = Xgalvo_gt(Xline_condition);
    x_1D  = x(Xline_condition);
    kx_1D = kx(Xline_condition);
    figure; plot(x_1D, Xgalvo_1D, '.-')
    figure; plot(kx_1D, Xgalvo_1D, '.-')
end



doplot = 0;
if doplot
    figure
    semilogy(Yslm_NRMSE, '.-')
    hold on
    semilogy(Xslm_NRMSE, '.-')
    semilogy(Ygalvo_NRMSE, '.-')
    semilogy(Xgalvo_NRMSE, '.-')
    hold off
    title(sprintf('STD-Normalized Root Mean Square Error\nTest set size: %.0f%%', test_percentage))
    legend({'Yslm', 'Xslm', 'Ygalvo', 'Xgalvo'}, 'Location', 'Best')
    xlabel('polynomial powers per dimension')
    ylabel('NRMSE')

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

    % Scatter plot kx & ky
    figure
    plot(cam_ft_row', cam_ft_col', '.')
    title('Fourier plane points')
    axis image
    xlabel('row coordinate (pix)')
    ylabel('column coordinate (pix)')

    % Scatter plot x & y
    figure
    plot(cam_img_row', cam_img_col', '.')
    title('Image plane points')
    axis image
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
end


function integers = map2integers(array)
    % Map a linear array of a perfect grid pattern coordinates to an array of indices
    % Array is assumed to be ordered according to a grid pattern. Majority of points are
    % assumed to be directly consequtive, as mode(diff(array)) is used to find the step size.
    %
    % For instance, arrays of 5 coordinates on a 2D plane, might be ordered according to the
    % following indices:
    %
    % y
    %  |  . 1 .        . 3 6
    %  |  2 3 4   or   1 4 7         where . is a point not present in the array
    %  |  . 5 .        2 5 .
    %  |_____________________ x
    %
    % Then this function will return the following integers in the same order:
    % 
    % y
    %  |  . 2 .        . 1 1
    %  |  1 2 3   or   2 2 2         where . is a point not present in the array
    %  |  . 2 .        3 3 .
    %  |_____________________ x

    % Difference array
    arraydiff = diff(array);
    
    % Find the most frequent stepsize in the array. This is assumed to be the grid step size.
    delta = mode(arraydiff(arraydiff > 0));
    
    % Map array to integers 1,2,3,...
    integers = round((array - min(array))/delta + 1);
end



function [A_grid] = lindex2grid(A, index_lin, sz)
    % Initialize NaN array of correct size
    A_grid = nan(sz);
    
    % Fill with values of input arrays, according to generated indices.
    A_grid(index_lin) = A;
end



% function [X_grid, Y_grid, index_lin, sz] = lin2grid2D(XY)
%     % Convert Nx2 array of grid point coordinates to 2D grid array, even if some of the points
%     % are missing. Note: Due to dependence on map2integers(), the majority of the points in the
%     % array are assumed to be directly consecutive. Output grid points not occurring in the
%     % input will be filled by NaNs.
%     %
%     % Input:
%     % XY                Nx2 num array containing the x- and y-coordinates of the grid points.
%     %
%     % Output:
%     % X_grid, Y_grid    2D arrays filled with the same points as the input. Output grid points 
%     %                   not occurring in the input will be filled by NaNs.
%     
%     % Unpack coordinate arrays
%     X = XY(:,1);
%     Y = XY(:,2);
%     
%     % Map arrays to appropriate grid indices, assuming the majority of the points in the array
%     % are directly consequtive.
%     index_X_grid = map2integers(X);
%     index_Y_grid = map2integers(Y);
% 
%     sz = [max(index_X_grid) max(index_Y_grid)];           % Size of 2D grid array
%     index_lin = sub2ind(sz, index_X_grid, index_Y_grid);  % Convert grid indices to linear indices
%     
%     [X_grid, Y_grid] = lindex2grid(X, Y, index_lin, sz);
% end







