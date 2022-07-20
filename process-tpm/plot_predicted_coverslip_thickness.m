clear; close all; clc

%% === Predicted coverslip thickness === %%
% Define found thicknesses
real_total_thickness_um = [1   2   3   4   5] * 170;
pred_total_thickness_um = [166 496 646 894 910];

real_total_thickness_2nd_um = [[1   2  3   4   5] * 170,  400];
pred_total_thickness_2nd_um = [ 367 42 559 786 868,       362];

% Subtract coverslip thickness for bottom objective
real_extra_thickness_um = real_total_thickness_um - 170;
pred_extra_thickness_um = pred_total_thickness_um - 170;
real_extra_thickness_2nd_um = real_total_thickness_2nd_um - 170;
pred_extra_thickness_2nd_um = pred_total_thickness_2nd_um - 170;

impossible_threshold = -5;
impossible_mask_2nd = (pred_extra_thickness_2nd_um < impossible_threshold);

% Plot
um = ['(' char(181) 'm)'];
figure; hold on
thickness_plot     = plot(real_extra_thickness_um,     pred_extra_thickness_um,     'xb', MarkerSize=16);
thickness_2nd_plot = plot(real_extra_thickness_2nd_um, pred_extra_thickness_2nd_um, 'xr', MarkerSize=16);
impossible_plot    = plot(real_extra_thickness_2nd_um(impossible_mask_2nd), pred_extra_thickness_2nd_um(impossible_mask_2nd), 'or', MarkerSize=16);
lin = linspace(0, max(real_extra_thickness_um), 2);
lin_plot = plot(lin, lin, 'k');
hold off
xlabel(['Real extra thickness ' um])
ylabel(['extra thickness ' um])
legend([thickness_plot thickness_2nd_plot impossible_plot lin_plot], {'Predicted 1st round', 'Predicted 2nd round', 'Impossible thickness', 'Equality'}, Location='SouthEast')
title(sprintf('Extra coverslip thickness\npredicted by Raylearn'))
set(gca, 'FontSize', 14)


%% === Magnification SLM to image cam === %%
% Define found thicknesses
std_magnification_SLM_to_imgcam =     [0.022 0.451 0.659 0.989 1.010];
std_magnification_SLM_to_imgcam_2nd = [0.274 0.179 0.541 0.848 0.954,  0.267];

% Subtract coverslip thickness for bottom objective
real_extra_thickness_um = real_total_thickness_um - 170;
real_extra_thickness_2nd_um = real_total_thickness_2nd_um - 170;

% Plot
um = ['(' char(181) 'm)'];
figure; hold on
thickness_plot     = plot(real_extra_thickness_um,     std_magnification_SLM_to_imgcam,     'xb', MarkerSize=16);
thickness_2nd_plot = plot(real_extra_thickness_2nd_um, std_magnification_SLM_to_imgcam_2nd, 'xr', MarkerSize=16);
% lin = linspace(0, max(real_extra_thickness_um), 2);
% lin_plot = plot(lin, lin, 'k');
hold off
xlabel(['Real extra thickness ' um])
ylabel('Magnification SLM to image cam')
legend([thickness_plot thickness_2nd_plot], {'1st round', '2nd round'}, Location='SouthEast')
title(sprintf('Magnification SLM to image cam\nvs extra coverslip thickness'))
set(gca, 'FontSize', 14)


%% === Correlation with predicted coverslip thickness === %
pred_extra_thickness_all_um = cat(2, pred_extra_thickness_um, pred_extra_thickness_2nd_um);
std_magnification_SLM_to_imgcam_all = cat(2, std_magnification_SLM_to_imgcam, std_magnification_SLM_to_imgcam_2nd);

fprintf('\nCorrelation of predicted thickness and SLM-imgcam magnification: %.2f%%\n', ...
    100 * corr(pred_extra_thickness_all_um', std_magnification_SLM_to_imgcam_all') ...
)

no_impossible_mask = (pred_extra_thickness_all_um > impossible_threshold);

fprintf('\nCorrelation of predicted thickness and SLM-imgcam magnification, without impossible thickness: %.2f%%\n', ...
    100 * corr(pred_extra_thickness_all_um(no_impossible_mask)', std_magnification_SLM_to_imgcam_all(no_impossible_mask)') ...
)







