%% Test to fit 4D polynomial, to model camera coords to SLM/Galvo coords
close all; clc; clear

load F:\ScientificData\raylearn-data\TPM\pencil-beam-positions\03-Nov-2020-empty\raylearn_pencil_beam_738098.472397_empty.mat

camsize_pix = 1024;
x  = cam_img_col(:) * 4/camsize_pix - 2;
y  = cam_img_row(:) * 4/camsize_pix - 2;
kx = cam_ft_col(:)  * 4/camsize_pix - 2;
ky = cam_ft_row(:)  * 4/camsize_pix - 2;

%% Create basis polynomial functions
% Create coordinate arrays for x,y,kx,ky

% Create 1D arrays containing 1, x, x^2, ..., 1, y, y^2, ..., 1, kx, kx^2, ...
npowers = 3;                                    % Polynomial powers (including 0)
orderlabels = cell(npowers.^4);
powers = 0:(npowers-1);                         % Array containing all powers

xpowers  =  x.^powers;
ypowers  =  y.^powers;
kxpowers = kx.^powers;
kypowers = ky.^powers;

Npoints = numel(cam_ft_col);
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


% Create test data from coefficients
Yslm_gt = repmat(p.rects(:,1), [size(cam_ft_col, 2) 1]);
Yslm_cf = xykxky_cam_basis \ Yslm_gt;                 % Compute coefficients
Yslm_fit = xykxky_cam_basis * Yslm_cf;                % Compute fit

Xslm_gt = repmat(p.rects(:,2), [size(cam_ft_row, 2) 1]);
Xslm_cf = xykxky_cam_basis \ Xslm_gt;                 % Compute coefficients
Xslm_fit = xykxky_cam_basis * Xslm_cf;                % Compute fit


% Plot slices of fit data vs ground truth
figure(1)
plot(kx, Xslm_gt, 'or')
hold on
plot(kx, Xslm_fit, '+b')
title('Fit')
xlabel('k_x (from cam)')
ylabel('X_{SLM} value')
legend('Ground Truth', 'Fit')
hold off


% figure(2)
% hbar = bar(Yslm_cf);    % Create bar plot
% title('Y_{SLM} coefficients')
% xlabel('index')
% ylabel('coefficient')
% % Get the data for all the bars that were plotted
% xbar = get(hbar,'XData');
% ybar = get(hbar,'YData');
% for i = 1:length(xbar) % Loop over each bar
%     if abs(Yslm_cf(i)) > 0.4*mean(abs(Yslm_cf))
%         htext = text(xbar(i),ybar(i),orders{i});          % Add text label
%         set(htext,'VerticalAlignment','middle',...  % Adjust properties
%                   'HorizontalAlignment','center')
%     end
% end
% 
% 
% figure(3)
% xfit  = linspace(min(x), max(x), 100);
% yfit  = mean(y);
% kxfit = mean(kx);
% kyfit = mean(ky);
% 
